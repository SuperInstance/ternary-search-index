# ternary-search-index

Ternary-weighted search index for GPU-accelerable document retrieval.

## Why This Exists

Standard TF-IDF uses floating-point weights. For ternary-weighted search, each term gets {-1, 0, +1}: negative demotes a document, positive promotes it, neutral has no effect. This compiles beautifully to XNOR + popcount on GPU — the same primitive used in ternary neural networks. You get a search index that runs at neural-network speed, and the ternary weights are interpretable: +1 means "this document is about this topic," -1 means "this document is anti-this-topic."

## Architecture

### Core Types

- **`TermWeight`** — Enum: `Negative (-1)`, `Neutral (0)`, `Positive (1)`.
- **`Document`** — A document with a map of term → weight.
- **`SearchResult`** — A scored hit: `doc_id`, `score` (sum of matching weights), `matches` (count).
- **`TernarySearchIndex`** — Collection of documents with `search()`, `batch_search()`, and `packed_score()`.

### Search Scoring

For a query Q and document D: `score = Σ Q[t] × D[t]` over all shared terms. Since values are {-1, 0, +1}, this is just XNOR + popcount in packed representation.

## Usage

```rust
use ternary_search_index::{Document, TernarySearchIndex, TermWeight};
use std::collections::HashMap;

let mut index = TernarySearchIndex::new();

let mut doc1 = Document::new(1);
doc1.set("gpu", TermWeight::Positive);
doc1.set("inference", TermWeight::Positive);
doc1.set("training", TermWeight::Negative);

let mut doc2 = Document::new(2);
doc2.set("gpu", TermWeight::Positive);
doc2.set("training", TermWeight::Positive);

index.add(doc1);
index.add(doc2);

let mut query = HashMap::new();
query.insert("gpu".into(), TermWeight::Positive);
query.insert("training".into(), TermWeight::Positive);

let results = index.search(&query, 10);
// doc2 scores higher (gpu=+1 + training=+1 = +2)
// doc1 scores lower (gpu=+1 + training=-1 = 0)
```

## API Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `Document::new(id)` | `Document` | Create empty document |
| `doc.set(term, weight)` | `()` | Set a term's ternary weight |
| `TernarySearchIndex::new()` | `TernarySearchIndex` | Create empty index |
| `index.add(doc)` | `()` | Index a document |
| `index.search(query, top_k)` | `Vec<SearchResult>` | Query for top-k matches |
| `TernarySearchIndex::packed_score(q, d)` | `i32` | Score packed vectors |
| `TernarySearchIndex::batch_search(q, docs, k)` | `Vec<(usize, i32)>` | Batch score |
| `index.doc_count()` | `usize` | Number of indexed documents |

## The Deeper Idea

Ternary search is **sentiment-aware information retrieval**. Traditional search treats term presence as positive signal. Ternary search lets you express "I want documents about GPUs but NOT about training" — a negative weight on "training" actively penalizes documents that are about training. This is equivalent to running a ternary classifier over your corpus where each dimension is a term and the query is the weight vector. The GPU parallelism comes free because the representation is identical to ternary neural network weights.

## Related Crates

- **ternary-bloom-filter** — membership testing with ternary weights
- **ternary-pack** — bit-packing for GPU efficiency
- **ternary-inference-sim** — simulated inference pipeline
