package main

import (
	"os"
	"path/filepath"
	"testing"
)

func testConfig() *Config {
	return &Config{
		IDField:    "_id",
		TextFields: []string{"title", "text"},
		MetaFields: []string{"category"},
	}
}

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return path
}

func drain(out chan Record) []Record {
	var recs []Record
	for r := range out {
		recs = append(recs, r)
	}
	return recs
}

func TestReadJSONL(t *testing.T) {
	content := `{"_id":"d1","title":"Hello","text":"world","category":"a"}
{"_id":"d2","title":"Foo","text":"bar"}
{broken json
{"_id":"","title":"no id","text":"skip me"}
`
	path := writeTemp(t, "docs.jsonl", content)
	out := make(chan Record, 10)
	count, err := readJSONL(path, testConfig(), out, 0)
	close(out)
	if err != nil {
		t.Fatalf("readJSONL: %v", err)
	}
	if count != 2 {
		t.Errorf("count: want 2, got %d", count)
	}
	recs := drain(out)
	if len(recs) != 2 {
		t.Fatalf("records: want 2, got %d", len(recs))
	}
	if recs[0].ID != "d1" || recs[0].Text != "Hello world" {
		t.Errorf("rec0: got %+v", recs[0])
	}
	if recs[0].Metadata["category"] != "a" {
		t.Errorf("rec0 metadata: got %+v", recs[0].Metadata)
	}
	if recs[1].ID != "d2" || recs[1].Text != "Foo bar" {
		t.Errorf("rec1: got %+v", recs[1])
	}
}

func TestReadJSONL_MaxRecords(t *testing.T) {
	content := `{"_id":"d1","title":"a","text":"1"}
{"_id":"d2","title":"b","text":"2"}
{"_id":"d3","title":"c","text":"3"}
`
	path := writeTemp(t, "docs.jsonl", content)
	out := make(chan Record, 10)
	count, err := readJSONL(path, testConfig(), out, 2)
	close(out)
	if err != nil {
		t.Fatalf("readJSONL: %v", err)
	}
	if count != 2 {
		t.Errorf("count: want 2, got %d", count)
	}
}

func TestReadJSONArray(t *testing.T) {
	content := `[{"_id":"a1","title":"first","text":"doc"},{"_id":"a2","title":"second","text":"doc"}]`
	path := writeTemp(t, "docs.json", content)
	out := make(chan Record, 10)
	count, err := readJSON(path, testConfig(), out, 0)
	close(out)
	if err != nil {
		t.Fatalf("readJSON: %v", err)
	}
	if count != 2 {
		t.Errorf("count: want 2, got %d", count)
	}
	recs := drain(out)
	if recs[0].ID != "a1" || recs[1].ID != "a2" {
		t.Errorf("ids: got %+v", recs)
	}
}

func TestReadJSONSingleObject(t *testing.T) {
	content := `{"_id":"s1","title":"single","text":"object"}`
	path := writeTemp(t, "doc.json", content)
	out := make(chan Record, 10)
	count, err := readJSON(path, testConfig(), out, 0)
	close(out)
	if err != nil {
		t.Fatalf("readJSON: %v", err)
	}
	if count != 1 {
		t.Fatalf("count: want 1, got %d", count)
	}
	recs := drain(out)
	if recs[0].ID != "s1" || recs[0].Text != "single object" {
		t.Errorf("rec: got %+v", recs[0])
	}
}

func TestReadCSV(t *testing.T) {
	content := "_id,title,text,category\n" +
		"c1,Alpha,body one,science\n" +
		"c2,Beta,body two,tech\n" +
		",Gamma,body three,tech\n" // missing id -> falls back to sequential index
	path := writeTemp(t, "docs.csv", content)
	out := make(chan Record, 10)
	count, err := readCSV(path, testConfig(), out, 0)
	close(out)
	if err != nil {
		t.Fatalf("readCSV: %v", err)
	}
	if count != 3 {
		t.Errorf("count: want 3, got %d", count)
	}
	recs := drain(out)
	if recs[0].ID != "c1" || recs[0].Text != "Alpha body one" {
		t.Errorf("rec0: got %+v", recs[0])
	}
	if recs[0].Metadata["category"] != "science" {
		t.Errorf("rec0 metadata: got %+v", recs[0].Metadata)
	}
	if recs[2].ID != "2" { // fell back to sequential doc index
		t.Errorf("rec2 fallback id: got %q", recs[2].ID)
	}
}

func TestMapToRecord_MissingID(t *testing.T) {
	_, err := mapToRecord([]byte(`{"title":"no id here","text":"body"}`), testConfig())
	if err == nil {
		t.Fatal("expected error for missing id field")
	}
}

func TestMapToRecord_AllTextFieldsEmpty(t *testing.T) {
	_, err := mapToRecord([]byte(`{"_id":"x1","title":"","text":""}`), testConfig())
	if err == nil {
		t.Fatal("expected error for empty text fields")
	}
}

func TestReadGen(t *testing.T) {
	cfg := &Config{GenSeed: 42}
	out := make(chan Record, 100)
	count, err := readGen("gen://50", cfg, out, 0)
	close(out)
	if err != nil {
		t.Fatalf("readGen: %v", err)
	}
	if count != 50 {
		t.Errorf("count: want 50, got %d", count)
	}
	recs := drain(out)
	if len(recs) != 50 {
		t.Fatalf("records: want 50, got %d", len(recs))
	}
	if recs[0].ID != "doc-0000000000" {
		t.Errorf("first id: got %q", recs[0].ID)
	}
	for _, r := range recs {
		if r.Text == "" {
			t.Error("expected non-empty generated text")
		}
	}
}

func TestReadGen_Deterministic(t *testing.T) {
	cfg := &Config{GenSeed: 7}
	out1 := make(chan Record, 20)
	readGen("gen://10", cfg, out1, 0)
	close(out1)
	recs1 := drain(out1)

	out2 := make(chan Record, 20)
	readGen("gen://10", cfg, out2, 0)
	close(out2)
	recs2 := drain(out2)

	for i := range recs1 {
		if recs1[i].Text != recs2[i].Text {
			t.Fatalf("same seed produced different text at index %d:\n%q\nvs\n%q", i, recs1[i].Text, recs2[i].Text)
		}
	}
}

func TestReadGen_InvalidURI(t *testing.T) {
	out := make(chan Record, 1)
	_, err := readGen("gen://not-a-number", &Config{}, out, 0)
	close(out)
	if err == nil {
		t.Fatal("expected error for invalid gen:// URI")
	}
}

func TestReadGen_MaxRecordsCapsN(t *testing.T) {
	out := make(chan Record, 100)
	count, err := readGen("gen://100", &Config{GenSeed: 1}, out, 10)
	close(out)
	if err != nil {
		t.Fatalf("readGen: %v", err)
	}
	if count != 10 {
		t.Errorf("count: want 10 (capped by maxRecords), got %d", count)
	}
}

func TestReadSourceFile_UnknownFormat(t *testing.T) {
	out := make(chan Record, 1)
	_, err := readSourceFile(sourceFile{path: "x", format: "xml"}, testConfig(), out, 0)
	close(out)
	if err == nil {
		t.Fatal("expected error for unknown format")
	}
}

func TestReadSourceFile_DispatchesJSONL(t *testing.T) {
	path := writeTemp(t, "d.jsonl", `{"_id":"j1","title":"t","text":"x"}`+"\n")
	out := make(chan Record, 1)
	count, err := readSourceFile(sourceFile{path: path, format: "jsonl"}, testConfig(), out, 0)
	close(out)
	if err != nil {
		t.Fatalf("readSourceFile: %v", err)
	}
	if count != 1 {
		t.Errorf("count: want 1, got %d", count)
	}
}
