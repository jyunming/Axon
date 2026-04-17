use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};

#[pyclass]
struct Bm25Index {
    postings: HashMap<String, Vec<(usize, u32)>>,
    doc_len: Vec<usize>,
    idf: HashMap<String, f64>,
    avgdl: f64,
    k1: f64,
    b: f64,
}

#[pyclass]
struct SymbolIndex {
    entries: Vec<(usize, String, String)>,
    exact: HashMap<String, Vec<(usize, String)>>,
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(|x| x.to_string())
        .collect()
}

#[pyfunction]
fn preprocess_documents(documents: &Bound<'_, PyList>, _batch_size: usize) -> PyResult<Py<PyList>> {
    let py = documents.py();
    let out = PyList::empty(py);
    for item in documents.iter() {
        let d = item.downcast::<PyDict>()?;
        let new_d = PyDict::new(py);
        for (k, v) in d.iter() {
            new_d.set_item(k, v)?;
        }
        if let Ok(Some(text_obj)) = d.get_item("text") {
            if let Ok(text) = text_obj.extract::<String>() {
                let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
                new_d.set_item("text", cleaned)?;
            }
        }
        out.append(new_d)?;
    }
    Ok(out.unbind())
}

#[pyfunction]
fn build_bm25_index(corpus: &Bound<'_, PyList>) -> PyResult<Bm25Index> {
    let mut postings: HashMap<String, Vec<(usize, u32)>> = HashMap::new();
    let mut doc_len: Vec<usize> = Vec::with_capacity(corpus.len());
    let mut dfs: HashMap<String, usize> = HashMap::new();
    let mut doc_idx: usize = 0;

    for item in corpus.iter() {
        let d = item.downcast::<PyDict>()?;
        let text = match d.get_item("text")? {
            Some(v) => v.extract::<String>().unwrap_or_default(),
            None => String::new(),
        };
        let toks = tokenize(&text);
        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for t in toks {
            use std::collections::hash_map::Entry;
            match tf_map.entry(t) {
                Entry::Vacant(v) => {
                    *dfs.entry(v.key().clone()).or_insert(0) += 1;
                    v.insert(1);
                }
                Entry::Occupied(mut o) => {
                    *o.get_mut() += 1;
                }
            }
        }
        for (term, tf) in tf_map {
            postings.entry(term).or_default().push((doc_idx, tf));
        }
        let dl = text.split_whitespace().count();
        doc_len.push(dl);
        doc_idx += 1;
    }

    let n = doc_len.len() as f64;
    let avgdl = if n > 0.0 {
        doc_len.iter().sum::<usize>() as f64 / n
    } else {
        0.0
    };
    let mut idf: HashMap<String, f64> = HashMap::new();
    for (term, df) in dfs {
        let df_f = df as f64;
        let v = ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln();
        idf.insert(term, v);
    }
    Ok(Bm25Index {
        postings,
        doc_len,
        idf,
        avgdl,
        k1: 1.5,
        b: 0.75,
    })
}

#[pyfunction]
fn search_bm25(index: &Bm25Index, query: &str, top_k: usize) -> Vec<(usize, f64)> {
    if index.doc_len.is_empty() {
        return vec![];
    }
    let q = tokenize(query);
    if q.is_empty() {
        return vec![];
    }
    let mut q_unique: Vec<String> = Vec::new();
    let mut q_seen: HashSet<String> = HashSet::new();
    for t in q {
        if q_seen.insert(t.clone()) {
            q_unique.push(t);
        }
    }

    let n_docs = index.doc_len.len();
    let mut scores_buf: Vec<f64> = vec![0.0; n_docs];
    let mut touched: Vec<usize> = Vec::new();
    for t in &q_unique {
        let Some(postings) = index.postings.get(t) else {
            continue;
        };
        let idf = *index.idf.get(t).unwrap_or(&0.0);
        if idf == 0.0 {
            continue;
        }
        for (doc_idx, freq) in postings {
            let i = *doc_idx;
            let dl = index.doc_len[i] as f64;
            let norm = index.k1 * (1.0 - index.b + index.b * dl / index.avgdl.max(1e-9));
            let tf_f = *freq as f64;
            let contrib = idf * ((tf_f * (index.k1 + 1.0)) / (tf_f + norm));
            if scores_buf[i] == 0.0 {
                touched.push(i);
            }
            scores_buf[i] += contrib;
        }
    }
    let mut scores: Vec<(usize, f64)> = touched
        .into_iter()
        .map(|i| (i, scores_buf[i]))
        .filter(|(_, s)| *s > 0.0)
        .collect();
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scores.truncate(top_k);
    scores
}

#[pyfunction(signature = (corpora, query_tokens, top_k, filters=None))]
fn symbol_channel_search(
    py: Python<'_>,
    corpora: &Bound<'_, PyList>,
    query_tokens: &Bound<'_, PyList>,
    top_k: usize,
    filters: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyList>> {
    let mut tokens: Vec<String> = Vec::new();
    for t in query_tokens.iter() {
        if let Ok(s) = t.extract::<String>() {
            if s.len() >= 3 {
                tokens.push(s.to_lowercase());
            }
        }
    }
    let out = PyList::empty(py);
    if tokens.is_empty() {
        return Ok(out.unbind());
    }
    let mut idx_global: usize = 0;
    let mut hits: Vec<(usize, f64, String)> = Vec::new();

    for corpus_any in corpora.iter() {
        let corpus = corpus_any.downcast::<PyList>()?;
        for doc_any in corpus.iter() {
            let doc = doc_any.downcast::<PyDict>()?;
            let Some(meta_any) = doc.get_item("metadata")? else {
                idx_global += 1;
                continue;
            };
            let meta = meta_any.downcast::<PyDict>()?;

            let passes_filters = if let Some(f) = filters {
                let mut ok = true;
                for (fk, fv) in f.iter() {
                    let Some(v) = meta.get_item(fk)? else {
                        ok = false;
                        break;
                    };
                    let a = v.str()?.to_string();
                    let b = fv.str()?.to_string();
                    if a != b {
                        ok = false;
                        break;
                    }
                }
                ok
            } else {
                true
            };
            if !passes_filters {
                idx_global += 1;
                continue;
            }

            let sym = meta
                .get_item("symbol_name")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
                .to_lowercase();
            let qn = meta
                .get_item("qualified_name")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
                .to_lowercase();
            if sym.is_empty() && qn.is_empty() {
                idx_global += 1;
                continue;
            }
            let mut best_score: f64 = 0.0;
            let mut channel = String::from("symbol_name");
            for tok in &tokens {
                if !sym.is_empty() {
                    if sym == *tok {
                        best_score = 1.0;
                        channel = String::from("symbol_name");
                    } else if best_score < 1.0 && sym.contains(tok) {
                        best_score = best_score.max(0.6);
                        channel = String::from("symbol_name");
                    }
                }
                if !qn.is_empty() {
                    if qn == *tok {
                        best_score = 1.0;
                        channel = String::from("qualified_name");
                    } else if best_score < 1.0 && qn.contains(tok) {
                        best_score = best_score.max(0.6);
                        channel = String::from("qualified_name");
                    }
                }
            }
            if best_score > 0.0 {
                hits.push((idx_global, best_score, channel));
            }
            idx_global += 1;
        }
    }

    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(top_k);
    for (idx, score, channel) in hits {
        let d = PyDict::new(py);
        d.set_item("index", idx)?;
        d.set_item("score", score)?;
        d.set_item("channel", channel)?;
        out.append(d)?;
    }
    Ok(out.unbind())
}

#[pyfunction]
fn build_symbol_index(corpora: &Bound<'_, PyList>) -> PyResult<SymbolIndex> {
    let mut idx_global: usize = 0;
    let mut entries: Vec<(usize, String, String)> = Vec::new();
    let mut exact: HashMap<String, Vec<(usize, String)>> = HashMap::new();

    for corpus_any in corpora.iter() {
        let corpus = corpus_any.downcast::<PyList>()?;
        for doc_any in corpus.iter() {
            let doc = doc_any.downcast::<PyDict>()?;
            let Some(meta_any) = doc.get_item("metadata")? else {
                idx_global += 1;
                continue;
            };
            let meta = meta_any.downcast::<PyDict>()?;
            let sym = meta
                .get_item("symbol_name")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
                .to_lowercase();
            let qn = meta
                .get_item("qualified_name")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
                .to_lowercase();
            if !sym.is_empty() || !qn.is_empty() {
                entries.push((idx_global, sym.clone(), qn.clone()));
                if !sym.is_empty() {
                    exact
                        .entry(sym)
                        .or_default()
                        .push((idx_global, String::from("symbol_name")));
                }
                if !qn.is_empty() {
                    exact
                        .entry(qn)
                        .or_default()
                        .push((idx_global, String::from("qualified_name")));
                }
            }
            idx_global += 1;
        }
    }
    Ok(SymbolIndex { entries, exact })
}

#[pyfunction]
fn search_symbol_index(
    index: &SymbolIndex,
    query_tokens: &Bound<'_, PyList>,
    top_k: usize,
) -> Vec<(usize, f64, String)> {
    let mut tokens: Vec<String> = Vec::new();
    for t in query_tokens.iter() {
        if let Ok(s) = t.extract::<String>() {
            if s.len() >= 3 {
                tokens.push(s.to_lowercase());
            }
        }
    }
    if tokens.is_empty() {
        return vec![];
    }

    let mut hits: HashMap<usize, (f64, String)> = HashMap::new();

    // Exact fast-path
    for tok in &tokens {
        if let Some(v) = index.exact.get(tok) {
            for (idx, ch) in v {
                hits.entry(*idx).or_insert((1.0, ch.clone()));
            }
        }
    }

    // Partial only if needed
    if hits.len() < top_k {
        for (idx, sym, qn) in &index.entries {
            if hits.contains_key(idx) {
                continue;
            }
            let mut best = 0.0;
            let mut ch = String::from("symbol_name");
            if !sym.is_empty() && tokens.iter().any(|t| sym.contains(t)) {
                best = 0.6;
                ch = String::from("symbol_name");
            }
            if !qn.is_empty() && tokens.iter().any(|t| qn.contains(t)) && best < 0.6 {
                best = 0.6;
                ch = String::from("qualified_name");
            }
            if best > 0.0 {
                hits.insert(*idx, (best, ch));
            }
        }
    }

    let mut out: Vec<(usize, f64, String)> = hits
        .into_iter()
        .map(|(idx, (score, ch))| (idx, score, ch))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(top_k);
    out
}

#[pymodule]
fn axon_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bm25Index>()?;
    m.add_class::<SymbolIndex>()?;
    m.add_function(wrap_pyfunction!(preprocess_documents, m)?)?;
    m.add_function(wrap_pyfunction!(build_bm25_index, m)?)?;
    m.add_function(wrap_pyfunction!(search_bm25, m)?)?;
    m.add_function(wrap_pyfunction!(symbol_channel_search, m)?)?;
    m.add_function(wrap_pyfunction!(build_symbol_index, m)?)?;
    m.add_function(wrap_pyfunction!(search_symbol_index, m)?)?;
    Ok(())
}
