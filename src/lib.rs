//! # ternary-search-index
//!
//! Ternary-weighted search index for GPU-accelerable retrieval.
//! Uses {-1, 0, +1} term weights: negative demotes, positive promotes.
//! Designed to compile to XNOR+popcount on GPU.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermWeight { Negative = -1, Neutral = 0, Positive = 1 }
impl TermWeight { pub fn val(&self) -> i32 { *self as i32 } }

#[derive(Debug, Clone)]
pub struct Document { pub id: u64, pub terms: HashMap<String, TermWeight> }
impl Document {
    pub fn new(id: u64) -> Self { Self { id, terms: HashMap::new() } }
    pub fn set(&mut self, term: &str, w: TermWeight) { self.terms.insert(term.into(), w); }
}

#[derive(Debug, Clone)]
pub struct SearchResult { pub doc_id: u64, pub score: i32, pub matches: usize }

pub struct TernarySearchIndex { docs: Vec<Document> }

impl TernarySearchIndex {
    pub fn new() -> Self { Self { docs: Vec::new() } }

    pub fn add(&mut self, doc: Document) { self.docs.push(doc); }

    pub fn search(&self, query: &HashMap<String, TermWeight>, top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self.docs.iter().map(|doc| {
            let mut score = 0i32;
            let mut matches = 0;
            for (term, qw) in query {
                if let Some(dw) = doc.terms.get(term) {
                    score += qw.val() * dw.val();
                    matches += 1;
                }
            }
            SearchResult { doc_id: doc.id, score, matches }
        }).collect();
        results.sort_by(|a, b| b.score.cmp(&a.score));
        results.truncate(top_k);
        results
    }

    /// GPU-packed search: ternary dot product over packed arrays.
    /// Simulates XNOR+popcount on GPU.
    pub fn packed_score(query: &[i8], doc: &[i8]) -> i32 {
        query.iter().zip(doc).map(|(q, d)| (*q as i32) * (*d as i32)).sum()
    }

    /// Pack document into i8 array for GPU transfer.
    pub fn pack_doc(doc: &Document, vocab: &[String]) -> Vec<i8> {
        vocab.iter().map(|t| doc.terms.get(t).map(|w| w.val() as i8).unwrap_or(0)).collect()
    }

    /// Batch GPU search: score multiple docs against query.
    pub fn batch_search(query: &[i8], docs: &[Vec<i8>], top_k: usize) -> Vec<(usize, i32)> {
        let mut scored: Vec<(usize, i32)> = docs.iter().enumerate()
            .map(|(i, d)| (i, Self::packed_score(query, d))).collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(top_k);
        scored
    }

    pub fn doc_count(&self) -> usize { self.docs.len() }
}

impl Default for TernarySearchIndex { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ranking() {
        let mut idx = TernarySearchIndex::new();
        let mut d1 = Document::new(1); d1.set("rust", TermWeight::Positive); d1.set("fast", TermWeight::Positive);
        let mut d2 = Document::new(2); d2.set("rust", TermWeight::Positive); d2.set("slow", TermWeight::Positive);
        idx.add(d1); idx.add(d2);
        let q = HashMap::from([("fast".into(), TermWeight::Positive)]);
        let r = idx.search(&q, 10);
        assert_eq!(r[0].doc_id, 1);
    }

    #[test]
    fn test_negative_demotion() {
        let mut idx = TernarySearchIndex::new();
        let mut d1 = Document::new(1); d1.set("rust", TermWeight::Positive);
        let mut d2 = Document::new(2); d2.set("rust", TermWeight::Negative);
        idx.add(d1); idx.add(d2);
        let q = HashMap::from([("rust".into(), TermWeight::Positive)]);
        let r = idx.search(&q, 10);
        assert_eq!(r[0].doc_id, 1); // positive*positive beats positive*negative
    }

    #[test]
    fn test_packed_score() {
        let score = TernarySearchIndex::packed_score(&[1, -1, 0], &[1, -1, 0]);
        assert_eq!(score, 2); // 1*1 + (-1)*(-1) + 0 = 2
    }

    #[test]
    fn test_packed_mismatch() {
        let score = TernarySearchIndex::packed_score(&[1, -1, 0], &[-1, 1, 0]);
        assert_eq!(score, -2); // 1*(-1) + (-1)*1 = -2
    }

    #[test]
    fn test_batch_search() {
        let query = vec![1, -1, 0];
        let docs = vec![vec![1, -1, 0], vec![-1, 1, 0], vec![0, 0, 1]];
        let results = TernarySearchIndex::batch_search(&query, &docs, 2);
        assert_eq!(results[0].0, 0); // best match
        assert_eq!(results[0].1, 2);
    }

    #[test]
    fn test_pack_doc() {
        let mut d = Document::new(1);
        d.set("a", TermWeight::Positive); d.set("b", TermWeight::Negative);
        let packed = TernarySearchIndex::pack_doc(&d, &["a".into(), "b".into(), "c".into()]);
        assert_eq!(packed, vec![1, -1, 0]);
    }

    #[test]
    fn test_empty_query() {
        let mut idx = TernarySearchIndex::new();
        idx.add(Document::new(1));
        let r = idx.search(&HashMap::new(), 10);
        assert_eq!(r[0].score, 0);
    }

    #[test]
    fn test_top_k() {
        let mut idx = TernarySearchIndex::new();
        for i in 0..10 { let mut d = Document::new(i); d.set("x", TermWeight::Positive); idx.add(d); }
        let r = idx.search(&HashMap::from([("x".into(), TermWeight::Positive)]), 3);
        assert_eq!(r.len(), 3);
    }
}
