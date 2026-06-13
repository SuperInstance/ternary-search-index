# Ternary Search Index — GPU-Accelerable Retrieval with Ternary Term Weights

**Ternary Search Index** is a search engine where documents and queries are represented as sparse vectors of ternary term weights: {-1 (demote), 0 (neutral), +1 (promote)}. The matching function is a ternary dot product that compiles to XNOR+popcount on GPU hardware, enabling high-throughput document ranking with O(1) per-term scoring.

## Why It Matters

Traditional search engines use TF-IDF or BM25 scoring with floating-point arithmetic — fast on CPU, but not GPU-friendly. Ternary search replaces float multiplications with sign comparisons: if the query promotes a term (+1) and the document has it (+1), the contribution is +1; if the query demotes it (-1) and the document has it, contribution is -1. This maps to a single XNOR+popcount instruction on GPU, processing 32 term comparisons per warp instruction. For billion-document indices, this means 10-50× faster ranking than float-based approaches, with minimal relevance loss.

## How It Works

### Document Model

Each `Document` is a sparse map of terms to `TermWeight` {-1, 0, +1}. The ternary value captures sentiment/relevance: +1 means the term is positively associated, -1 means negatively associated, 0 means the term appears but is neutral.

### Scoring

The search score is a ternary dot product:

```
score(query, doc) = Σ query[term] × doc[term]   over all terms
```

Since each multiplication is between ternary values, the result per term is in {-1, 0, +1}. The sum is an integer in [-k, +k] for k matching terms. This is O(k) per document-query pair.

### GPU Packing

For GPU execution, documents are packed into `Vec<i8>` arrays indexed by vocabulary position. The query is similarly packed. Scoring becomes:

```
packed_score(query: &[i8], doc: &[i8]) = Σ query[i] × doc[i]
```

On GPU, this compiles to XNOR (for sign agreement) + popcount (to count agreements), processing 32 elements per instruction.

### Batch Search

`batch_search(query, docs, top_k)` scores multiple packed documents against a packed query, returning the top-k indices. O(n × d) for n documents of dimension d, embarrassingly parallel across documents.

### Ranking

Results are sorted by score (descending) and truncated to top_k. Documents with negative scores are demoted; positive scores are promoted.

## Quick Start

```rust
use ternary_search_index::{TernarySearchIndex, Document, TermWeight};

let mut idx = TernarySearchIndex::new();

let mut doc1 = Document::new(1);
doc1.set("rust", TermWeight::Positive);
doc1.set("fast", TermWeight::Positive);
doc1.set("slow", TermWeight::Negative);

let mut doc2 = Document::new(2);
doc2.set("rust", TermWeight::Positive);
doc2.set("slow", TermWeight::Positive);

idx.add(doc1);
idx.add(doc2);

// Search
let mut query = std::collections::HashMap::new();
query.insert("rust".into(), TermWeight::Positive);
query.insert("fast".into(), TermWeight::Positive);

let results = idx.search(&query, 10);
println!("Top result: doc {}", results[0].doc_id);
```

```bash
cargo add ternary-search-index
```

## API

| Type / Function | Description |
|---|---|
| `TermWeight` | `Negative(-1)`, `Neutral(0)`, `Positive(1)` |
| `Document` | Sparse term→weight map: `set()`, `terms` |
| `TernarySearchIndex` | `add()`, `search()`, `batch_search()`, `packed_score()` |
| `SearchResult` | `{ doc_id, score, matches }` |

## Architecture Notes

This is the retrieval layer for **SuperInstance** knowledge management. Fleet agents use ternary search to find relevant documents, code, and configurations. The γ + η = C conservation manifests in the scoring: positive matches contribute γ (relevant content), negative matches contribute η (irrelevant content), and the net score is bounded by total term count C. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Manning, Christopher et al. *Introduction to Information Retrieval*, Cambridge UP, 2008 — IR fundamentals.
- Rastegari, Mohammad et al. "XNOR-Net," *ECCV*, 2016 — binary/ternary operations on GPU.
- Robertson, Stephen & Zaragoza, Hugo. "The Probabilistic Relevance Framework: BM25 and Beyond," *Found. Trends IR*, 3(4), 2009.

## License

Apache-2.0
