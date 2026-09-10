#!/usr/bin/env python3
"""
Benchmark harness: ingests MSMARCO into each search system and measures
effectiveness (nDCG@10, MRR@10, Recall@100) and efficiency (latency, QPS).

Usage:
  # With Docker systems running:
  docker compose -f bench/docker-compose.bench.yml up -d
  python3 bench/run_bench.py [--systems all] [--skip-ingest] [--top-k 100]

  # Quick effectiveness-only run (no ingest, server must be pre-loaded):
  python3 bench/run_bench.py --skip-ingest --systems search-engine

Prerequisites:
  pip install requests tqdm pyarrow huggingface_hub numpy
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

# Config

SYSTEMS = {
    "search-engine": {
        "base_url": "http://localhost:8080",
        "ingest_fn": "ingest_search_engine",
        "search_fn": "search_search_engine",
        "label": "search-engine (ours)",
    },
    "elasticsearch": {
        "base_url": "http://localhost:9200",
        "ingest_fn": "ingest_elasticsearch",
        "search_fn": "search_elasticsearch",
        "label": "Elasticsearch 8.17",
    },
    "meilisearch": {
        "base_url": "http://localhost:7700",
        "ingest_fn": "ingest_meilisearch",
        "search_fn": "search_meilisearch",
        "label": "Meilisearch v1.12",
    },
    "typesense": {
        "base_url": "http://localhost:8108",
        "ingest_fn": "ingest_typesense",
        "search_fn": "search_typesense",
        "label": "Typesense 27.1",
    },
    "quickwit": {
        "base_url": "http://localhost:7280",
        "ingest_fn": "ingest_quickwit",
        "search_fn": "search_quickwit",
        "label": "Quickwit 0.8.1",
    },
}

INDEX_NAME = "msmarco"
TYPESENSE_API_KEY = "bench-api-key"

# Data loading

def load_corpus(local_dir=None):
    """Load MSMARCO corpus. Returns iterator of {id, text} dicts."""
    if local_dir and Path(local_dir).exists():
        return _load_corpus_local(local_dir)
    return _load_corpus_hf()

def _load_corpus_hf():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    print("Downloading MSMARCO corpus from HuggingFace...")
    corpus_path = hf_hub_download(
        repo_id="BeIR/msmarco", repo_type="dataset",
        filename="corpus/corpus-00000-of-00001.parquet",
    )
    table = pq.read_table(corpus_path)
    for i in range(table.num_rows):
        yield {
            "id": str(table["_id"][i].as_py()),
            "text": str(table["text"][i].as_py()),
            "title": str(table["title"][i].as_py() or ""),
        }

def _load_corpus_local(local_dir):
    import pyarrow.parquet as pq
    for f in sorted(Path(local_dir).glob("*.parquet")):
        table = pq.read_table(f)
        for i in range(table.num_rows):
            yield {
                "id": str(table["_id"][i].as_py()),
                "text": str(table["text"][i].as_py()),
                "title": str(table["title"][i].as_py() or ""),
            }

def _limit_iter(it, n):
    """Yield at most n items from iterator it."""
    count = 0
    for item in it:
        if count >= n:
            break
        yield item
        count += 1

def load_queries_and_qrels():
    """Load MSMARCO dev queries + qrels. Returns (queries, qrels)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print("Downloading MSMARCO queries and qrels...")

    # Queries (all splits combined; filter by qrels below)
    queries_path = hf_hub_download(
        repo_id="BeIR/msmarco", repo_type="dataset",
        filename="queries/queries-00000-of-00001.parquet",
    )
    qt = pq.read_table(queries_path)
    queries = {}
    for i in range(qt.num_rows):
        qid = str(qt["_id"][i].as_py())
        text = str(qt["text"][i].as_py())
        queries[qid] = text

    # Qrels from mteb/msmarco (dev split)
    import csv
    qrels_path = hf_hub_download(
        repo_id="mteb/msmarco", repo_type="dataset",
        filename="qrels/dev.tsv",
    )
    qrels = {}
    with open(qrels_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(row["score"])
            qrels.setdefault(qid, {})[doc_id] = score

    # Filter to only queries with qrels
    filtered = {qid: text for qid, text in queries.items() if qid in qrels}
    print(f"  Queries with qrels: {len(filtered)}")
    return filtered, qrels

# Ingest functions

def ingest_search_engine(base_url, docs, batch_size=500):
    """Ingest documents into our search engine via /index/bulk."""
    s = requests.Session()
    buf = []
    ingested = 0
    for doc in docs:
        text = doc["text"]
        if doc.get("title"):
            text = doc["title"] + " " + text
        buf.append(json.dumps({"id": doc["id"], "text": text}))
        if len(buf) >= batch_size:
            ndjson = "\n".join(buf)
            s.post(f"{base_url}/index/bulk", data=ndjson,
                   headers={"Content-Type": "application/x-ndjson"}).raise_for_status()
            ingested += len(buf)
            buf = []
    if buf:
        ndjson = "\n".join(buf)
        s.post(f"{base_url}/index/bulk", data=ndjson,
               headers={"Content-Type": "application/x-ndjson"}).raise_for_status()
        ingested += len(buf)
    # Flush and merge
    s.post(f"{base_url}/index/flush").raise_for_status()
    s.post(f"{base_url}/index/merge").raise_for_status()
    return ingested

def ingest_elasticsearch(base_url, docs, batch_size=2000):
    """Ingest documents into Elasticsearch via bulk API."""
    s = requests.Session()
    # Create index with BM25 settings
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}},
        },
        "mappings": {"properties": {"text": {"type": "text"}, "title": {"type": "text"}}},
    }
    r = s.put(f"{base_url}/{INDEX_NAME}", json=mapping)
    if r.status_code not in (200, 400):  # 400 = already exists
        r.raise_for_status()

    buf = []
    ingested = 0
    for doc in docs:
        buf.append(json.dumps({"index": {"_id": doc["id"]}}))
        buf.append(json.dumps({"text": doc["text"], "title": doc.get("title", "")}))
        if len(buf) >= batch_size * 2:
            body = "\n".join(buf) + "\n"
            r = s.post(f"{base_url}/{INDEX_NAME}/_bulk",
                       data=body, headers={"Content-Type": "application/x-ndjson"})
            r.raise_for_status()
            if r.json().get("errors"):
                raise RuntimeError(f"ES bulk errors: {r.text[:500]}")
            ingested += len(buf) // 2
            buf = []
    if buf:
        body = "\n".join(buf) + "\n"
        s.post(f"{base_url}/{INDEX_NAME}/_bulk",
               data=body, headers={"Content-Type": "application/x-ndjson"}).raise_for_status()
        ingested += len(buf) // 2
    s.post(f"{base_url}/{INDEX_NAME}/_refresh").raise_for_status()
    return ingested

def ingest_meilisearch(base_url, docs, batch_size=1000):
    """Ingest documents into Meilisearch."""
    s = requests.Session()
    # Create/configure index
    s.post(f"{base_url}/indexes", json={"uid": INDEX_NAME, "primaryKey": "id"})
    # Disable stop words for fair BM25 comparison
    s.patch(f"{base_url}/indexes/{INDEX_NAME}/settings",
            json={"stopWords": [], "rankingRules": ["words", "typo", "proximity", "attribute", "sort", "exactness"]})

    buf = []
    ingested = 0
    for doc in docs:
        buf.append({"id": doc["id"], "text": doc["text"], "title": doc.get("title", "")})
        if len(buf) >= batch_size:
            r = s.post(f"{base_url}/indexes/{INDEX_NAME}/documents", json=buf)
            r.raise_for_status()
            task_uid = r.json()["taskUid"]
            _wait_meilisearch_task(s, base_url, task_uid)
            ingested += len(buf)
            buf = []
    if buf:
        r = s.post(f"{base_url}/indexes/{INDEX_NAME}/documents", json=buf)
        r.raise_for_status()
        task_uid = r.json()["taskUid"]
        _wait_meilisearch_task(s, base_url, task_uid)
        ingested += len(buf)
    return ingested

def _wait_meilisearch_task(s, base_url, task_uid, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = s.get(f"{base_url}/tasks/{task_uid}")
        status = r.json().get("status")
        if status == "succeeded":
            return
        if status == "failed":
            raise RuntimeError(f"Meilisearch task {task_uid} failed: {r.json()}")
        time.sleep(0.5)
    raise TimeoutError(f"Meilisearch task {task_uid} timed out")

def ingest_typesense(base_url, docs, batch_size=1000):
    """Ingest documents into Typesense."""
    s = requests.Session()
    s.headers["X-TYPESENSE-API-KEY"] = TYPESENSE_API_KEY
    # Create collection
    schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "text", "type": "string"},
            {"name": "title", "type": "string", "optional": True},
        ],
        "default_sorting_field": "",
    }
    r = s.post(f"{base_url}/collections", json=schema)
    if r.status_code not in (200, 201, 409):  # 409 = already exists
        r.raise_for_status()

    buf = []
    ingested = 0
    for doc in docs:
        buf.append({"id": doc["id"], "text": doc["text"], "title": doc.get("title", "")})
        if len(buf) >= batch_size:
            ndjson = "\n".join(json.dumps(d) for d in buf)
            r = s.post(f"{base_url}/collections/{INDEX_NAME}/documents/import?action=upsert",
                       data=ndjson, headers={"Content-Type": "text/plain"})
            r.raise_for_status()
            ingested += len(buf)
            buf = []
    if buf:
        ndjson = "\n".join(json.dumps(d) for d in buf)
        s.post(f"{base_url}/collections/{INDEX_NAME}/documents/import?action=upsert",
               data=ndjson, headers={"Content-Type": "text/plain"}).raise_for_status()
        ingested += len(buf)
    return ingested

def ingest_quickwit(base_url, docs, batch_size=5000):
    """Ingest documents into Quickwit."""
    s = requests.Session()
    # Create index
    index_config = {
        "version": "0.8",
        "index_id": INDEX_NAME,
        "doc_mapping": {
            "mode": "dynamic",
            "field_mappings": [
                {"name": "id", "type": "text", "tokenizer": "raw", "indexed": True, "stored": True, "fast": True},
                {"name": "text", "type": "text", "tokenizer": "default", "indexed": True, "stored": True},
                {"name": "title", "type": "text", "tokenizer": "default", "indexed": True, "stored": True},
            ],
        },
        "search_settings": {"default_search_fields": ["text", "title"]},
    }
    r = s.post(f"{base_url}/api/v1/indexes", json=index_config)
    if r.status_code not in (200, 400):
        r.raise_for_status()

    buf = []
    ingested = 0
    for doc in docs:
        buf.append({"id": doc["id"], "text": doc["text"], "title": doc.get("title", "")})
        if len(buf) >= batch_size:
            ndjson = "\n".join(json.dumps(d) for d in buf)
            r = s.post(f"{base_url}/api/v1/{INDEX_NAME}/ingest?commit=auto",
                       data=ndjson, headers={"Content-Type": "application/json"})
            r.raise_for_status()
            ingested += len(buf)
            buf = []
    if buf:
        ndjson = "\n".join(json.dumps(d) for d in buf)
        # commit=force on the last batch ensures all docs are visible before search
        s.post(f"{base_url}/api/v1/{INDEX_NAME}/ingest?commit=force",
               data=ndjson, headers={"Content-Type": "application/json"}).raise_for_status()
        ingested += len(buf)
    elif ingested > 0:
        # flush if last batch was exactly batch_size (already sent with commit=auto)
        s.post(f"{base_url}/api/v1/{INDEX_NAME}/ingest?commit=force",
               data="{}", headers={"Content-Type": "application/json"})
    return ingested

# Search functions

def search_search_engine(base_url, query, top_k=100):
    r = requests.get(f"{base_url}/search",
                     params={"q": query, "top_k": top_k, "no_snippet": "1"},
                     timeout=30)
    r.raise_for_status()
    data = r.json()
    return [hit["doc_id"] for hit in data.get("results", [])]

def search_elasticsearch(base_url, query, top_k=100):
    body = {"query": {"match": {"text": query}}, "size": top_k, "_source": False}
    r = requests.post(f"{base_url}/{INDEX_NAME}/_search", json=body, timeout=30)
    r.raise_for_status()
    return [hit["_id"] for hit in r.json()["hits"]["hits"]]

def search_meilisearch(base_url, query, top_k=100):
    r = requests.post(f"{base_url}/indexes/{INDEX_NAME}/search",
                      json={"q": query, "limit": top_k, "attributesToRetrieve": ["id"]},
                      timeout=30)
    r.raise_for_status()
    return [hit["id"] for hit in r.json().get("hits", [])]

def search_typesense(base_url, query, top_k=100):
    params = {
        "q": query, "query_by": "text,title",
        "per_page": top_k, "include_fields": "id",
    }
    r = requests.get(f"{base_url}/collections/{INDEX_NAME}/documents/search",
                     params=params,
                     headers={"X-TYPESENSE-API-KEY": TYPESENSE_API_KEY},
                     timeout=30)
    r.raise_for_status()
    return [hit["document"]["id"] for hit in r.json().get("hits", [])]

def search_quickwit(base_url, query, top_k=100):
    params = {"query": query, "max_hits": top_k, "format": "json"}
    r = requests.get(f"{base_url}/api/v1/{INDEX_NAME}/search", params=params, timeout=30)
    r.raise_for_status()
    return [hit["id"] for hit in r.json().get("hits", [])]

# Metrics

def ndcg_at_k(results, qrels, k=10):
    """Compute nDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(results[:k]):
        rel = qrels.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)
    # Ideal DCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(results, qrels, k=10):
    """Compute MRR@k."""
    for i, doc_id in enumerate(results[:k]):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(results, qrels, k=100):
    """Compute Recall@k."""
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    retrieved = set(results[:k])
    return len(relevant & retrieved) / len(relevant)

def map_at_k(results, qrels, k=10):
    """Compute MAP@k."""
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    hits = 0
    ap = 0.0
    for i, doc_id in enumerate(results[:k]):
        if doc_id in relevant:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(relevant)

# Benchmark runner

def run_queries(system_name, system_cfg, queries, qrels, top_k, workers):
    """Run all queries against a system; return per-query metrics + latencies."""
    base_url = system_cfg["base_url"]
    search_fn = globals()[system_cfg["search_fn"]]

    results = []

    def run_one(args):
        qid, query = args
        t0 = time.perf_counter()
        try:
            hits = search_fn(base_url, query, top_k)
        except Exception as e:
            return {"qid": qid, "error": str(e), "latency_ms": 0,
                    "ndcg10": 0, "mrr10": 0, "recall100": 0, "map10": 0}
        latency_ms = (time.perf_counter() - t0) * 1000
        q_qrels = qrels.get(qid, {})
        return {
            "qid": qid,
            "latency_ms": latency_ms,
            "ndcg10": ndcg_at_k(hits, q_qrels, 10),
            "mrr10": mrr_at_k(hits, q_qrels, 10),
            "recall100": recall_at_k(hits, q_qrels, 100),
            "map10": map_at_k(hits, q_qrels, 10),
        }

    items = list(queries.items())
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, item): item for item in items}
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"  querying {system_name}"):
            results.append(f.result())
    elapsed = time.perf_counter() - t_start

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  WARNING: {len(errors)} query errors (first: {errors[0]['error']})")

    latencies = [r["latency_ms"] for r in results if "error" not in r]
    return {
        "ndcg10": np.mean([r["ndcg10"] for r in results]),
        "mrr10": np.mean([r["mrr10"] for r in results]),
        "recall100": np.mean([r["recall100"] for r in results]),
        "map10": np.mean([r["map10"] for r in results]),
        "mean_latency_ms": np.mean(latencies) if latencies else 0,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0,
        "p99_ms": float(np.percentile(latencies, 99)) if latencies else 0,
        "qps": len(results) / elapsed if elapsed > 0 else 0,
        "total_queries": len(results),
        "errors": len(errors),
    }

def wait_healthy(base_url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(base_url, timeout=3)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

# Index size helpers

def get_index_size_bytes(system_name, base_url):
    """Return approximate on-disk index size in bytes, or None if unavailable."""
    try:
        if system_name == "search-engine":
            r = requests.get(f"{base_url}/segments", timeout=10)
            r.raise_for_status()
            segs = r.json().get("segments", [])
            total = sum(s.get("size_bytes", 0) for s in segs)
            return total if total > 0 else None

        elif system_name == "elasticsearch":
            r = requests.get(f"{base_url}/{INDEX_NAME}/_stats/store", timeout=10)
            r.raise_for_status()
            return r.json()["_all"]["primaries"]["store"]["size_in_bytes"]

        elif system_name == "meilisearch":
            r = requests.get(f"{base_url}/indexes/{INDEX_NAME}/stats", timeout=10)
            r.raise_for_status()
            return r.json().get("databaseSize")

        elif system_name in ("typesense", "quickwit"):
            return None
    except Exception:
        return None
    return None

# Main

def main():
    parser = argparse.ArgumentParser(description="MSMARCO benchmark harness")
    parser.add_argument("--systems", default="all",
                        help="comma-separated systems to benchmark (default: all)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="skip ingestion (systems must be pre-loaded)")
    parser.add_argument("--corpus-dir", default=None,
                        help="local corpus parquet directory (optional)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="documents to retrieve per query (default: 100)")
    parser.add_argument("--workers", type=int, default=2,
                        help="concurrent query workers (default: 2)")
    parser.add_argument("--sample", type=int, default=0,
        help="Randomly sample N queries from the qrels set for a quick eval (0 = all 6980)")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit corpus to first N docs for smoke tests (0 = no limit)")
    parser.add_argument("--output", default="bench_results.json",
                        help="output JSON file for results")
    args = parser.parse_args()

    if args.systems == "all":
        selected = list(SYSTEMS.keys())
    else:
        selected = [s.strip() for s in args.systems.split(",")]
    selected = [s for s in selected if s in SYSTEMS]
    if not selected:
        print(f"No valid systems selected. Available: {list(SYSTEMS.keys())}")
        sys.exit(1)

    print(f"\nBenchmark: MSMARCO dev, top_k={args.top_k}, workers={args.workers}")
    print(f"Systems:   {selected}\n")

    # Load queries and qrels once
    queries, qrels = load_queries_and_qrels()
    if args.sample > 0 and args.sample < len(queries):
        import random
        sampled_ids = random.sample(list(queries.keys()), args.sample)
        queries = {qid: queries[qid] for qid in sampled_ids}
        print(f"Sampled {len(queries)} / 6980 queries (--sample {args.sample})\n")
    else:
        print(f"Loaded {len(queries)} queries with qrels\n")

    all_results = {}

    for system_name in selected:
        cfg = SYSTEMS[system_name]
        print(f"{'='*60}")
        print(f"System: {cfg['label']}")
        print(f"{'='*60}")

        # Health check
        health_url = cfg["base_url"]
        if system_name == "search-engine":
            health_url += "/health"
        elif system_name == "elasticsearch":
            health_url += "/_cluster/health"
        elif system_name == "meilisearch":
            health_url += "/health"
        elif system_name == "typesense":
            health_url += "/health"
        elif system_name == "quickwit":
            health_url += "/api/v1/version"

        if not wait_healthy(health_url, timeout=60):
            print(f"  SKIP: {system_name} not healthy at {cfg['base_url']}")
            continue

        # Ingest
        if not args.skip_ingest:
            print(f"  Ingesting corpus...")
            ingest_fn = globals()[cfg["ingest_fn"]]
            corpus = load_corpus(args.corpus_dir)
            if args.limit > 0:
                corpus = _limit_iter(corpus, args.limit)
            t0 = time.time()
            try:
                n = ingest_fn(cfg["base_url"], corpus)
            except Exception as e:
                print(f"  ERROR during ingest: {e}")
                continue
            ingest_elapsed = time.time() - t0
            print(f"  Ingested {n:,} docs in {ingest_elapsed:.1f}s ({n/ingest_elapsed:.0f} docs/s)")
        else:
            print(f"  Skipping ingest (--skip-ingest)")
            ingest_elapsed = None

        # Query
        print(f"  Running {len(queries)} queries...")
        metrics = run_queries(system_name, cfg, queries, qrels, args.top_k, args.workers)

        # Print results
        print(f"\n  Results ({cfg['label']}):")
        print(f"    nDCG@10:     {metrics['ndcg10']:.4f}")
        print(f"    MRR@10:      {metrics['mrr10']:.4f}")
        print(f"    Recall@100:  {metrics['recall100']:.4f}")
        print(f"    MAP@10:      {metrics['map10']:.4f}")
        print(f"    Mean latency:{metrics['mean_latency_ms']:.1f} ms")
        print(f"    p50:         {metrics['p50_ms']:.1f} ms")
        print(f"    p95:         {metrics['p95_ms']:.1f} ms")
        print(f"    QPS:         {metrics['qps']:.2f}")
        idx_bytes = get_index_size_bytes(system_name, cfg["base_url"])
        if idx_bytes is not None:
            print(f"    Index size:  {idx_bytes / (1024**3):.2f} GB")
        print()

        all_results[system_name] = {
            "label": cfg["label"],
            "ingest_secs": ingest_elapsed,
            "index_size_bytes": idx_bytes,
            **metrics,
        }

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*100}")
        print(f"{'System':<28} {'nDCG@10':>8} {'MRR@10':>8} {'R@100':>8} {'MAP@10':>8} "
              f"{'p50ms':>7} {'p95ms':>7} {'QPS':>7} {'IdxGB':>7}")
        print(f"{'-'*100}")
        for name, r in all_results.items():
            idx_str = f"{r['index_size_bytes']/(1024**3):.2f}" if r.get("index_size_bytes") else "  n/a"
            print(f"{r['label']:<28} {r['ndcg10']:>8.4f} {r['mrr10']:>8.4f} "
                  f"{r['recall100']:>8.4f} {r['map10']:>8.4f} "
                  f"{r['p50_ms']:>7.1f} {r['p95_ms']:>7.1f} {r['qps']:>7.2f} {idx_str:>7}")
        print(f"{'='*100}\n")

    # Save JSON results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
