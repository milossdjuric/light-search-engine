#!/usr/bin/env python3
"""
BEIR multi-dataset evaluation harness for our search engine.

For each requested BEIR dataset this script:
  1. Resets the server index via POST /admin/reset
  2. Downloads and ingests the corpus via /index/bulk
  3. Flushes + triggers merge
  4. Downloads qrels + queries, runs eval
  5. Prints per-dataset metrics and a final summary table

Usage:
  source bench/.venv/bin/activate
  python3 bench/run_beir.py [--datasets nfcorpus,fiqa,scifact] [--skip-ingest] \
                             [--workers N] [--top-k 100] [--output FILE]

Datasets available (corpus size):
  nfcorpus     3.6K docs   323 queries  medical nutrition
  scifact      5.2K docs   300 queries  scientific claim verification
  arguana      8.7K docs   467 queries  argument retrieval
  scidocs      25.6K docs  1000 queries scientific paper retrieval
  fiqa         57.6K docs  648  queries financial QA
  trec-covid   171K docs   50   queries COVID biomedical
  quora        523K docs   10K  queries duplicate questions
  nq           2.68M docs  3452 queries Natural Questions (Wikipedia)

Notes:
  - All BEIR datasets use the same HF schema: _id / title / text
  - Qrels live in BeIR/<name>-qrels repo (test split), except msmarco (dev)
  - trec-covid qrels use the 'test' split with partial judgments
"""
import argparse
import csv
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

# Dataset registry
# Each entry: hf_corpus_repo, qrels_repo, qrels_split, approx docs, approx queries
BEIR_DATASETS = {
    "nfcorpus": {
        "corpus_repo":  "BeIR/nfcorpus",
        "qrels_repo":   "BeIR/nfcorpus-qrels",
        "qrels_split":  "test",
        "approx_docs":  3_633,
        "approx_queries": 323,
        "description":  "Medical nutrition documents",
    },
    "scifact": {
        "corpus_repo":  "BeIR/scifact",
        "qrels_repo":   "BeIR/scifact-qrels",
        "qrels_split":  "test",
        "approx_docs":  5_183,
        "approx_queries": 300,
        "description":  "Scientific claim verification",
    },
    "arguana": {
        "corpus_repo":  "BeIR/arguana",
        "qrels_repo":   "BeIR/arguana-qrels",
        "qrels_split":  "test",
        "approx_docs":  8_674,
        "approx_queries": 467,
        "description":  "Argument retrieval (counter-arguments)",
    },
    "scidocs": {
        "corpus_repo":  "BeIR/scidocs",
        "qrels_repo":   "BeIR/scidocs-qrels",
        "qrels_split":  "test",
        "approx_docs":  25_657,
        "approx_queries": 1_000,
        "description":  "Scientific paper retrieval",
    },
    "fiqa": {
        "corpus_repo":  "BeIR/fiqa",
        "qrels_repo":   "BeIR/fiqa-qrels",
        "qrels_split":  "test",
        "approx_docs":  57_638,
        "approx_queries": 648,
        "description":  "Financial QA (Stack Exchange)",
    },
    "trec-covid": {
        "corpus_repo":  "BeIR/trec-covid",
        "qrels_repo":   "BeIR/trec-covid-qrels",
        "qrels_split":  "test",
        "approx_docs":  171_332,
        "approx_queries": 50,
        "description":  "Biomedical COVID literature",
    },
    "quora": {
        "corpus_repo":  "BeIR/quora",
        "qrels_repo":   "BeIR/quora-qrels",
        "qrels_split":  "test",
        "approx_docs":  522_931,
        "approx_queries": 10_000,
        "description":  "Duplicate question retrieval",
    },
    "nq": {
        "corpus_repo":  "BeIR/nq",
        "qrels_repo":   "BeIR/nq-qrels",
        "qrels_split":  "test",
        "approx_docs":  2_681_468,
        "approx_queries": 3_452,
        "description":  "Natural Questions (Wikipedia)",
    },
    "hotpotqa": {
        "corpus_repo":  "BeIR/hotpotqa",
        "qrels_repo":   "BeIR/hotpotqa-qrels",
        "qrels_split":  "test",
        "approx_docs":  5_233_329,
        "approx_queries": 7_405,
        "description":  "Multi-hop Wikipedia QA",
    },
    "msmarco": {
        "corpus_repo":  "BeIR/msmarco",
        "qrels_repo":   "mteb/msmarco",        # different repo for MSMARCO
        "qrels_split":  "dev",                  # dev split for MSMARCO
        "approx_docs":  8_841_823,
        "approx_queries": 6_980,
        "description":  "MS MARCO passage retrieval",
    },
}

DEFAULT_DATASETS = "nfcorpus,scifact,arguana,fiqa,trec-covid"
SERVER_URL = "http://localhost:8080"
ES_URL = "http://localhost:9200"

# Data loading

def load_corpus(dataset_name):
    """Download corpus parquet(s) from HF and yield {id, title, text} dicts."""
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq

    cfg = BEIR_DATASETS[dataset_name]
    repo = cfg["corpus_repo"]
    print(f"  Downloading corpus from {repo}...")

    # Find all corpus parquet files (may be sharded)
    files = [
        f for f in list_repo_files(repo, repo_type="dataset")
        if f.startswith("corpus/") and f.endswith(".parquet")
    ]
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No corpus parquet files found in {repo}")

    for filepath in files:
        local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filepath)
        table = pq.read_table(local)
        for i in range(table.num_rows):
            yield {
                "id":    str(table["_id"][i].as_py()),
                "title": str(table["title"][i].as_py() or ""),
                "text":  str(table["text"][i].as_py()),
            }


def load_queries_and_qrels(dataset_name):
    """Download queries + qrels from HF. Returns (queries dict, qrels dict)."""
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq

    cfg = BEIR_DATASETS[dataset_name]
    corpus_repo = cfg["corpus_repo"]
    qrels_repo  = cfg["qrels_repo"]
    split       = cfg["qrels_split"]

    # Queries
    print(f"  Downloading queries from {corpus_repo}...")
    q_files = sorted(
        f for f in list_repo_files(corpus_repo, repo_type="dataset")
        if f.startswith("queries/") and f.endswith(".parquet")
    )
    if not q_files:
        raise RuntimeError(f"No query parquet files in {corpus_repo}")

    queries = {}
    for filepath in q_files:
        local = hf_hub_download(repo_id=corpus_repo, repo_type="dataset", filename=filepath)
        table = pq.read_table(local)
        for i in range(table.num_rows):
            qid  = str(table["_id"][i].as_py())
            text = str(table["text"][i].as_py())
            queries[qid] = text

    # Qrels
    print(f"  Downloading qrels from {qrels_repo} ({split})...")
    # Try <split>.tsv (most datasets) or qrels/<split>.tsv (mteb layout)
    qrels = {}
    fetched = False
    for candidate in [f"{split}.tsv", f"qrels/{split}.tsv"]:
        try:
            local = hf_hub_download(repo_id=qrels_repo, repo_type="dataset", filename=candidate)
            with open(local) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    qid    = str(row["query-id"])
                    doc_id = str(row["corpus-id"])
                    score  = int(row["score"])
                    qrels.setdefault(qid, {})[doc_id] = score
            fetched = True
            break
        except Exception:
            continue

    if not fetched:
        raise RuntimeError(f"Could not download qrels for {dataset_name} from {qrels_repo}")

    # Filter queries to only those that have qrels
    filtered = {qid: text for qid, text in queries.items() if qid in qrels}
    print(f"  Queries with qrels: {len(filtered)}")
    return filtered, qrels

# Ingest

def reset_index(server_url):
    """Wipe the search engine index via POST /admin/reset."""
    r = requests.post(f"{server_url}/admin/reset", timeout=30)
    r.raise_for_status()

def ingest_corpus(server_url, corpus_iter, batch_size=500, show_progress=True):
    """Ingest docs via /index/bulk, then flush + merge. Returns doc count."""
    s = requests.Session()
    buf = []
    ingested = 0
    t0 = time.time()

    it = tqdm(corpus_iter, desc="  ingesting", unit="doc") if show_progress else corpus_iter
    for doc in it:
        text = doc["text"]
        if doc.get("title"):
            text = doc["title"] + " " + text
        buf.append(json.dumps({"id": doc["id"], "text": text}))
        if len(buf) >= batch_size:
            ndjson = "\n".join(buf)
            s.post(f"{server_url}/index/bulk", data=ndjson,
                   headers={"Content-Type": "application/x-ndjson"}).raise_for_status()
            ingested += len(buf)
            buf = []

    if buf:
        ndjson = "\n".join(buf)
        s.post(f"{server_url}/index/bulk", data=ndjson,
               headers={"Content-Type": "application/x-ndjson"}).raise_for_status()
        ingested += len(buf)

    elapsed = time.time() - t0
    print(f"  Indexed {ingested:,} docs in {elapsed:.1f}s ({ingested/elapsed:.0f} docs/s)")

    print("  Flushing + merging...")
    s.post(f"{server_url}/index/flush").raise_for_status()
    s.post(f"{server_url}/index/merge").raise_for_status()

    # Wait for merge to settle (poll /health until warm=true)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            h = requests.get(f"{server_url}/health", timeout=5).json()
            if h.get("warm"):
                break
        except Exception:
            pass
        time.sleep(2)

    return ingested

# Elasticsearch ingest + search

def reset_es(es_url, index_name):
    """Delete the Elasticsearch index if it exists."""
    requests.delete(f"{es_url}/{index_name}", timeout=10)

def ingest_es(es_url, index_name, corpus_iter, batch_size=2000, show_progress=True):
    """Ingest docs into Elasticsearch via bulk API. Returns doc count."""
    s = requests.Session()
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",       # disable auto-refresh during ingest
            "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}},
        },
        "mappings": {"properties": {
            "text":  {"type": "text", "analyzer": "standard"},
            "title": {"type": "text", "analyzer": "standard"},
        }},
    }
    r = s.put(f"{es_url}/{index_name}", json=mapping)
    if r.status_code not in (200, 400):
        r.raise_for_status()

    buf = []
    ingested = 0
    t0 = time.time()
    it = tqdm(corpus_iter, desc="  [ES] ingesting", unit="doc") if show_progress else corpus_iter
    for doc in it:
        text = doc.get("title", "")
        if text:
            text += " " + doc["text"]
        else:
            text = doc["text"]
        buf.append(json.dumps({"index": {"_id": doc["id"]}}))
        buf.append(json.dumps({"text": text}))
        if len(buf) >= batch_size * 2:
            body = "\n".join(buf) + "\n"
            r = s.post(f"{es_url}/{index_name}/_bulk",
                       data=body.encode(), headers={"Content-Type": "application/x-ndjson"},
                       timeout=120)
            r.raise_for_status()
            ingested += len(buf) // 2
            buf = []
    if buf:
        body = "\n".join(buf) + "\n"
        s.post(f"{es_url}/{index_name}/_bulk",
               data=body.encode(), headers={"Content-Type": "application/x-ndjson"},
               timeout=120).raise_for_status()
        ingested += len(buf) // 2
    # Re-enable refresh and force a final refresh
    s.put(f"{es_url}/{index_name}/_settings",
          json={"index": {"refresh_interval": "1s"}}).raise_for_status()
    s.post(f"{es_url}/{index_name}/_refresh", timeout=60).raise_for_status()
    elapsed = time.time() - t0
    print(f"  [ES] Indexed {ingested:,} docs in {elapsed:.1f}s ({ingested/elapsed:.0f} docs/s)")
    return ingested

def search_es(es_url, index_name, query, top_k, timeout=30):
    """Query Elasticsearch and return ordered list of doc IDs."""
    body = {
        "query": {"multi_match": {"query": query, "fields": ["text", "title"]}},
        "size": top_k,
        "_source": False,
    }
    r = requests.post(f"{es_url}/{index_name}/_search", json=body, timeout=timeout)
    r.raise_for_status()
    return [hit["_id"] for hit in r.json()["hits"]["hits"]]

def run_eval_es(es_url, index_name, queries, qrels, top_k, workers):
    """Same as run_eval but against Elasticsearch."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    items = list(queries.items())
    t_start = time.perf_counter()

    def run_one(args):
        qid, query = args
        t0 = time.perf_counter()
        try:
            hits = search_es(es_url, index_name, query, top_k)
        except Exception as e:
            return {"qid": qid, "error": str(e), "latency_ms": 0,
                    "ndcg10": 0.0, "mrr10": 0.0, "recall100": 0.0}
        latency_ms = (time.perf_counter() - t0) * 1000
        q_qrels = qrels.get(qid, {})
        return {
            "qid": qid,
            "latency_ms": latency_ms,
            "ndcg10":    ndcg_at_k(hits, q_qrels, 10),
            "mrr10":     mrr_at_k(hits, q_qrels, 10),
            "recall100": recall_at_k(hits, q_qrels, 100),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, item): item for item in items}
        for f in tqdm(as_completed(futs), total=len(futs), desc="  [ES] querying"):
            results.append(f.result())

    elapsed = time.perf_counter() - t_start
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  [ES] WARNING: {len(errors)} query errors")
    latencies = [r["latency_ms"] for r in results if "error" not in r]
    return {
        "ndcg10":        float(np.mean([r["ndcg10"]    for r in results])),
        "mrr10":         float(np.mean([r["mrr10"]     for r in results])),
        "recall100":     float(np.mean([r["recall100"] for r in results])),
        "p50_ms":        float(np.percentile(latencies, 50))  if latencies else 0.0,
        "p95_ms":        float(np.percentile(latencies, 95))  if latencies else 0.0,
        "qps":           len(results) / elapsed if elapsed > 0 else 0.0,
        "total_queries": len(results),
        "errors":        len(errors),
    }

# Search

def search(server_url, query, top_k, timeout=30):
    r = requests.get(f"{server_url}/search",
                     params={"q": query, "top_k": top_k, "no_snippet": "1"},
                     timeout=timeout)
    r.raise_for_status()
    return [hit["doc_id"] for hit in r.json().get("results", [])]

# Metrics

def ndcg_at_k(results, qrels, k=10):
    dcg = sum(
        (2**qrels.get(doc, 0) - 1) / math.log2(i + 2)
        for i, doc in enumerate(results[:k])
    )
    idcg = sum(
        (2**r - 1) / math.log2(i + 2)
        for i, r in enumerate(sorted(qrels.values(), reverse=True)[:k])
    )
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(results, qrels, k=10):
    for i, doc in enumerate(results[:k]):
        if qrels.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(results, qrels, k=100):
    relevant = {d for d, r in qrels.items() if r > 0}
    return len(relevant & set(results[:k])) / len(relevant) if relevant else 0.0

# Eval runner

def run_eval(server_url, queries, qrels, top_k, workers):
    items = list(queries.items())
    t_start = time.perf_counter()

    def run_one(args):
        qid, query = args
        t0 = time.perf_counter()
        try:
            hits = search(server_url, query, top_k)
        except Exception as e:
            return {"qid": qid, "error": str(e), "latency_ms": 0,
                    "ndcg10": 0.0, "mrr10": 0.0, "recall100": 0.0}
        latency_ms = (time.perf_counter() - t0) * 1000
        q_qrels = qrels.get(qid, {})
        return {
            "qid": qid,
            "latency_ms": latency_ms,
            "ndcg10":    ndcg_at_k(hits, q_qrels, 10),
            "mrr10":     mrr_at_k(hits, q_qrels, 10),
            "recall100": recall_at_k(hits, q_qrels, 100),
        }

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, item): item for item in items}
        for f in tqdm(as_completed(futs), total=len(futs), desc="  querying"):
            results.append(f.result())

    elapsed = time.perf_counter() - t_start
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  WARNING: {len(errors)} query errors (first: {errors[0]['error']})")

    latencies = [r["latency_ms"] for r in results if "error" not in r]
    return {
        "ndcg10":         float(np.mean([r["ndcg10"]    for r in results])),
        "mrr10":          float(np.mean([r["mrr10"]     for r in results])),
        "recall100":      float(np.mean([r["recall100"] for r in results])),
        "p50_ms":         float(np.percentile(latencies, 50))  if latencies else 0.0,
        "p95_ms":         float(np.percentile(latencies, 95))  if latencies else 0.0,
        "qps":            len(results) / elapsed if elapsed > 0 else 0.0,
        "total_queries":  len(results),
        "errors":         len(errors),
    }

# Main

def main():
    parser = argparse.ArgumentParser(description="BEIR multi-dataset evaluation")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS,
                        help=f"comma-separated dataset names (default: {DEFAULT_DATASETS})")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="skip reset+ingest; eval against whatever is currently in the index")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent query workers (default: 4)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="top-K docs to retrieve per query (default: 100)")
    parser.add_argument("--server", default=SERVER_URL,
                        help=f"search engine base URL (default: {SERVER_URL})")
    parser.add_argument("--output", default="bench/beir_results.json",
                        help="JSON output file (default: bench/beir_results.json)")
    parser.add_argument("--compare-es", action="store_true",
                        help="also run the same eval against Elasticsearch and print a comparison table")
    parser.add_argument("--es-url", default=ES_URL,
                        help=f"Elasticsearch base URL (default: {ES_URL})")
    parser.add_argument("--es-skip-ingest", action="store_true",
                        help="skip ES ingest (dataset already loaded in ES)")
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    unknown = [d for d in dataset_names if d not in BEIR_DATASETS]
    if unknown:
        print(f"Unknown datasets: {unknown}")
        print(f"Available: {list(BEIR_DATASETS.keys())}")
        sys.exit(1)

    print(f"\nBEIR evaluation — datasets: {dataset_names}")
    print(f"Server: {args.server} | top_k={args.top_k} | workers={args.workers}")
    if args.compare_es:
        print(f"Elasticsearch: {args.es_url}")
    print()

    # BEIR BM25 published baselines (nDCG@10 from the BEIR paper, BM25 Pyserini)
    BM25_BASELINES = {
        "nfcorpus":   0.321, "scifact":  0.665, "arguana":   0.472,
        "scidocs":    0.158, "fiqa":     0.236, "trec-covid": 0.656,
        "quora":      0.789, "nq":       0.329, "hotpotqa":  0.603,
        "msmarco":    0.228,
    }

    all_results = {}

    for dataset_name in dataset_names:
        cfg = BEIR_DATASETS[dataset_name]
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name} — {cfg['description']}")
        print(f"  Corpus ≈{cfg['approx_docs']:,} docs | Queries ≈{cfg['approx_queries']:,}")
        print(f"{'='*60}")

        # Load corpus into memory once (shared between our engine and ES).
        corpus_docs = None

        if not args.skip_ingest:
            print("  Loading corpus from HuggingFace...")
            try:
                corpus_docs = list(load_corpus(dataset_name))
            except Exception as e:
                print(f"  ERROR loading corpus: {e}")
                import traceback; traceback.print_exc()
                continue

            # Reset + ingest our engine
            print("  Resetting our engine index...")
            try:
                reset_index(args.server)
            except Exception as e:
                print(f"  WARN: reset failed ({e}), continuing anyway")
            try:
                ingest_corpus(args.server, iter(corpus_docs))
            except Exception as e:
                print(f"  ERROR during ingest: {e}")
                import traceback; traceback.print_exc()
                continue

            # Reset + ingest Elasticsearch (if compare mode)
            if args.compare_es and not args.es_skip_ingest:
                index_name = dataset_name.replace("-", "_")
                print(f"  Resetting ES index '{index_name}'...")
                try:
                    reset_es(args.es_url, index_name)
                except Exception as e:
                    print(f"  WARN: ES reset failed ({e}), continuing anyway")
                try:
                    ingest_es(args.es_url, index_name, iter(corpus_docs))
                except Exception as e:
                    print(f"  ERROR during ES ingest: {e}")
                    import traceback; traceback.print_exc()
        else:
            print("  Skipping ingest (--skip-ingest)")

        # Load queries + qrels
        try:
            queries, qrels = load_queries_and_qrels(dataset_name)
        except Exception as e:
            print(f"  ERROR loading queries/qrels: {e}")
            import traceback; traceback.print_exc()
            continue

        # Eval — our engine
        print(f"\n  [Our engine] Running {len(queries)} queries (workers={args.workers})...")
        metrics = run_eval(args.server, queries, qrels, args.top_k, args.workers)

        # Eval — Elasticsearch
        es_metrics = None
        if args.compare_es:
            index_name = dataset_name.replace("-", "_")
            print(f"  [ES] Running {len(queries)} queries (workers={args.workers})...")
            try:
                es_metrics = run_eval_es(args.es_url, index_name, queries, qrels,
                                         args.top_k, args.workers)
            except Exception as e:
                print(f"  ERROR during ES eval: {e}")

        baseline = BM25_BASELINES.get(dataset_name)

        if es_metrics:
            # Side-by-side comparison
            print(f"\n  {'':20s} {'Our engine':>12} {'Elasticsearch':>14} {'delta':>7}")
            print(f"  {'-'*56}")
            for label, key, fmt in [
                ("nDCG@10",    "ndcg10",    ".4f"),
                ("MRR@10",     "mrr10",     ".4f"),
                ("Recall@100", "recall100", ".4f"),
                ("p50 (ms)",   "p50_ms",    ".1f"),
                ("p95 (ms)",   "p95_ms",    ".1f"),
                ("QPS",        "qps",       ".1f"),
            ]:
                ours = metrics[key]
                theirs = es_metrics[key]
                delta = ours - theirs
                print(f"  {label:<20s} {ours:>12{fmt}} {theirs:>14{fmt}} {delta:>+7.3f}" if "ms" not in label and "QPS" not in label
                      else f"  {label:<20s} {ours:>12{fmt}} {theirs:>14{fmt}}")
            if baseline:
                print(f"\n  Pyserini BM25 baseline (nDCG@10): {baseline:.4f}")
        else:
            delta_str = ""
            if baseline:
                delta = metrics["ndcg10"] - baseline
                delta_str = f"  (BM25 baseline {baseline:.3f}, delta {delta:+.3f})"
            print(f"\n  Results ({dataset_name}):")
            print(f"    nDCG@10:    {metrics['ndcg10']:.4f}{delta_str}")
            print(f"    MRR@10:     {metrics['mrr10']:.4f}")
            print(f"    Recall@100: {metrics['recall100']:.4f}")
            print(f"    p50:        {metrics['p50_ms']:.1f} ms")
            print(f"    p95:        {metrics['p95_ms']:.1f} ms")
            print(f"    QPS:        {metrics['qps']:.1f}")
        print()

        all_results[dataset_name] = {
            "description":   cfg["description"],
            "approx_docs":   cfg["approx_docs"],
            "bm25_baseline": baseline,
            **{f"ours_{k}": v for k, v in metrics.items()},
            **({"es_" + k: v for k, v in es_metrics.items()} if es_metrics else {}),
        }

    # Summary table
    if all_results:
        has_es = any("es_ndcg10" in r for r in all_results.values())
        if has_es:
            print(f"\n{'='*110}")
            print(f"{'BEIR Summary — Our Engine vs Elasticsearch':}")
            print(f"{'='*110}")
            print(f"{'Dataset':<15} {'Docs':>8} "
                  f"{'Ours nDCG':>10} {'ES nDCG':>9} {'Δ nDCG':>7} "
                  f"{'Ours R@100':>11} {'ES R@100':>9} "
                  f"{'Ours QPS':>9} {'ES QPS':>8} "
                  f"{'vs BM25':>8}")
            print(f"{'-'*110}")
            for name, r in all_results.items():
                baseline = r.get("bm25_baseline")
                delta_ndcg = r.get("ours_ndcg10", 0) - r.get("es_ndcg10", 0)
                delta_vs = f"{r.get('ours_ndcg10', 0) - baseline:+.3f}" if baseline else "  n/a"
                print(f"{name:<15} {r['approx_docs']:>8,} "
                      f"{r.get('ours_ndcg10', 0):>10.4f} {r.get('es_ndcg10', 0):>9.4f} {delta_ndcg:>+7.3f} "
                      f"{r.get('ours_recall100', 0):>11.4f} {r.get('es_recall100', 0):>9.4f} "
                      f"{r.get('ours_qps', 0):>9.1f} {r.get('es_qps', 0):>8.1f} "
                      f"{delta_vs:>8}")
            print(f"{'='*110}\n")
            ours_mean = np.mean([r.get("ours_ndcg10", 0) for r in all_results.values()])
            es_mean   = np.mean([r.get("es_ndcg10",   0) for r in all_results.values()])
            print(f"Macro-avg nDCG@10 — Ours: {ours_mean:.4f}  |  ES: {es_mean:.4f}  |  Δ: {ours_mean - es_mean:+.4f}")
            print()
        else:
            print(f"\n{'='*90}")
            print(f"{'BEIR Summary':}")
            print(f"{'='*90}")
            print(f"{'Dataset':<15} {'Docs':>10} {'nDCG@10':>9} {'MRR@10':>9} {'R@100':>8} "
                  f"{'p50ms':>7} {'QPS':>7} {'vs BM25':>9}")
            print(f"{'-'*90}")
            for name, r in all_results.items():
                baseline = r.get("bm25_baseline")
                delta_str = f"{r.get('ours_ndcg10', r.get('ndcg10', 0)) - baseline:+.3f}" if baseline else "  n/a"
                ndcg = r.get("ours_ndcg10", r.get("ndcg10", 0))
                mrr  = r.get("ours_mrr10",  r.get("mrr10",  0))
                rec  = r.get("ours_recall100", r.get("recall100", 0))
                p50  = r.get("ours_p50_ms", r.get("p50_ms", 0))
                qps  = r.get("ours_qps",    r.get("qps",    0))
                print(f"{name:<15} {r['approx_docs']:>10,} {ndcg:>9.4f} "
                      f"{mrr:>9.4f} {rec:>8.4f} "
                      f"{p50:>7.1f} {qps:>7.1f} {delta_str:>9}")
            print(f"{'='*90}\n")
            macro_ndcg = np.mean([r.get("ours_ndcg10", r.get("ndcg10", 0)) for r in all_results.values()])
            print(f"Macro-avg nDCG@10 across {len(all_results)} datasets: {macro_ndcg:.4f}")
            print()

    # Save JSON
    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
