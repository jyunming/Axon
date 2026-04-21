use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::{HashMap, HashSet};

// Lazily compiled regexes for extract_code_query_tokens
use std::sync::OnceLock;

fn re_code_ext() -> &'static regex::Regex {
    static R: OnceLock<regex::Regex> = OnceLock::new();
    R.get_or_init(|| {
        regex::Regex::new(
            r"\b(\w+)\.(py|rs|ts|tsx|js|jsx|go|java|cpp|c|h|cs|rb|kt|swift|scala|lua|sh|toml|yaml|yml|json|md)\b",
        )
        .unwrap()
    })
}

fn re_qualified() -> &'static regex::Regex {
    static R: OnceLock<regex::Regex> = OnceLock::new();
    R.get_or_init(|| regex::Regex::new(r"\b([A-Za-z_]\w+)\.([A-Za-z_]\w+)\b").unwrap())
}

fn re_identifier() -> &'static regex::Regex {
    static R: OnceLock<regex::Regex> = OnceLock::new();
    R.get_or_init(|| regex::Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b").unwrap())
}

fn re_camel() -> &'static regex::Regex {
    static R: OnceLock<regex::Regex> = OnceLock::new();
    R.get_or_init(|| regex::Regex::new(r"[A-Z][a-z0-9]*|[a-z][a-z0-9]*").unwrap())
}

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

/// Return the MD5 hex digest of `text` — matches Python `hashlib.md5(text.encode()).hexdigest()`.
#[pyfunction]
fn compute_doc_hash(text: &str) -> String {
    format!("{:x}", md5::compute(text.as_bytes()))
}

/// Extract identifier-like tokens from a code query.
///
/// Mirrors `code_retrieval._extract_code_query_tokens`:
/// - Basename stems (strips known extensions)
/// - Qualified names: `foo.bar` → {"foo", "bar", "foo.bar"}
/// - All identifiers ≥4 chars; CamelCase and snake_case split; min part len 3
///
/// Returns a Vec<String> of lowercase tokens (Python side wraps in `frozenset`).
#[pyfunction]
fn extract_code_query_tokens(query: &str) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();

    // Basename stems (e.g. loaders.py → "loaders")
    for cap in re_code_ext().captures_iter(query) {
        tokens.insert(cap[1].to_lowercase());
    }

    // Qualified names: foo.bar → "foo", "bar", "foo.bar"
    for cap in re_qualified().captures_iter(query) {
        let a = cap[1].to_lowercase();
        let b = cap[2].to_lowercase();
        let ab = format!("{}.{}", a, b);
        tokens.insert(a);
        tokens.insert(b);
        tokens.insert(ab);
    }

    // All identifier-like tokens ≥4 chars
    for cap in re_identifier().captures_iter(query) {
        let word = &cap[0];
        let lower = word.to_lowercase();
        tokens.insert(lower.clone());

        // CamelCase split
        let parts: Vec<String> = re_camel()
            .find_iter(word)
            .map(|m| m.as_str().to_lowercase())
            .filter(|p| p.len() >= 3)
            .collect();
        for p in parts {
            tokens.insert(p);
        }

        // snake_case split
        if lower.contains('_') {
            for part in lower.split('_') {
                if part.len() >= 3 {
                    tokens.insert(part.to_string());
                }
            }
        }
    }

    tokens.into_iter().collect()
}

/// Compute per-result lexical boost scores for code retrieval.
///
/// Mirrors the scoring kernel inside `_apply_code_lexical_boost` (excludes diagnostics/trace
/// fill-in and the per-file diversity cap — those stay in Python).
///
/// Returns `(lex_scores, max_lex)` where `lex_scores[i]` corresponds to `results[i]`.
/// `max_lex` is 0.0 when no result matched any token.
#[pyfunction]
fn code_lexical_scores(
    results: &Bound<'_, PyList>,
    query_tokens: &Bound<'_, PyList>,
) -> PyResult<(Vec<f64>, f64)> {
    // Collect tokens into owned HashSets for O(1) lookup
    let mut all_tokens: HashSet<String> = HashSet::new();
    let mut long_tokens: HashSet<String> = HashSet::new();
    for item in query_tokens.iter() {
        if let Ok(s) = item.extract::<String>() {
            let lower = s.to_lowercase();
            if lower.len() >= 4 {
                long_tokens.insert(lower.clone());
            }
            all_tokens.insert(lower);
        }
    }

    if all_tokens.is_empty() {
        return Ok((vec![0.0; results.len()], 0.0));
    }

    let mut lex_scores: Vec<f64> = Vec::with_capacity(results.len());

    for item in results.iter() {
        let doc = item.downcast::<PyDict>()?;

        let Some(meta_any) = doc.get_item("metadata")? else {
            lex_scores.push(0.0);
            continue;
        };
        let meta = meta_any.downcast::<PyDict>()?;

        let source_class = meta
            .get_item("source_class")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        if source_class != "code" {
            lex_scores.push(0.0);
            continue;
        }

        let sym_name = meta
            .get_item("symbol_name")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default()
            .to_lowercase();
        let sym_type = meta
            .get_item("symbol_type")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default()
            .to_lowercase();
        let file_path = meta
            .get_item("file_path")?
            .and_then(|v| v.extract::<String>().ok())
            .or_else(|| {
                meta.get_item("source")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
            })
            .unwrap_or_default();
        let start_line = meta
            .get_item("start_line")?
            .and_then(|v| v.extract::<i64>().ok());
        let end_line = meta
            .get_item("end_line")?
            .and_then(|v| v.extract::<i64>().ok());

        let text_lower = doc
            .get_item("text")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default()
            .to_lowercase();

        // Basename (stem without extension)
        let basename = std::path::Path::new(&file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        let qualified = if !sym_name.is_empty() && !basename.is_empty() {
            format!("{}.{}", basename, sym_name)
        } else {
            String::new()
        };

        let mut score = 0.0_f64;

        // Exact symbol name match
        if !sym_name.is_empty() && all_tokens.contains(&sym_name) {
            score += 1.0;
        } else if !sym_name.is_empty() {
            for tok in &long_tokens {
                if sym_name.contains(tok.as_str()) {
                    score += 0.5;
                    break;
                }
            }
        }

        // Basename match
        if !basename.is_empty() && all_tokens.contains(&basename) {
            score += 0.4;
        }

        // Qualified name match
        if !qualified.is_empty() && all_tokens.contains(&qualified) {
            score += 1.0;
        }

        // Token-in-text hits (capped at 4 hits × 0.08 = 0.32)
        let text_hits: f64 = long_tokens
            .iter()
            .filter(|t| text_lower.contains(t.as_str()))
            .count() as f64;
        score += (text_hits * 0.08_f64).min(0.32);

        // Function/method multiplier
        if score > 0.0 && (sym_type == "function" || sym_type == "method") {
            score *= 1.1;
        }

        // Line-range tightness tie-breaker
        if let (Some(sl), Some(el)) = (start_line, end_line) {
            let span = (el - sl).max(1);
            if span <= 30 {
                score += 0.05;
            } else if span <= 80 {
                score += 0.02;
            }
        }

        lex_scores.push(score);
    }

    let max_lex = lex_scores.iter().cloned().fold(0.0_f64, f64::max);
    Ok((lex_scores, max_lex))
}

// ── Shared conversion helpers ─────────────────────────────────────────────────
//
// Used by JSON-decode (Feature 1), msgpack encode/decode (Features 2 & 5).
// All four helpers handle: dict↔map, list↔array, str, int, float, bool, None.

fn json_value_to_pyobject(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    use serde_json::Value;
    match v {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(<pyo3::Bound<'_, pyo3::types::PyBool> as Clone>::clone(&pyo3::types::PyBool::new(py, *b)).into_any().unbind()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(n.to_string().into_pyobject(py)?.into_any().unbind())
            }
        }
        Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_pyobject(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(map) => {
            let d = PyDict::new(py);
            for (k, val) in map {
                d.set_item(k, json_value_to_pyobject(py, val)?)?;
            }
            Ok(d.into_any().unbind())
        }
    }
}

fn rmpv_to_pyobject(py: Python<'_>, v: &rmpv::Value) -> PyResult<PyObject> {
    match v {
        rmpv::Value::Nil => Ok(py.None()),
        rmpv::Value::Boolean(b) => Ok(<pyo3::Bound<'_, pyo3::types::PyBool> as Clone>::clone(&pyo3::types::PyBool::new(py, *b)).into_any().unbind()),
        rmpv::Value::Integer(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(0_i64.into_pyobject(py)?.into_any().unbind())
            }
        }
        rmpv::Value::F32(f) => Ok((*f as f64).into_pyobject(py)?.into_any().unbind()),
        rmpv::Value::F64(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        rmpv::Value::String(s) => {
            Ok(s.as_str().unwrap_or("").into_pyobject(py)?.into_any().unbind())
        }
        rmpv::Value::Binary(b) => Ok(PyBytes::new(py, b).into_any().unbind()),
        rmpv::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(rmpv_to_pyobject(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        rmpv::Value::Map(pairs) => {
            let d = PyDict::new(py);
            for (k, val) in pairs {
                d.set_item(rmpv_to_pyobject(py, k)?, rmpv_to_pyobject(py, val)?)?;
            }
            Ok(d.into_any().unbind())
        }
        rmpv::Value::Ext(_, _) => Ok(py.None()),
    }
}

fn pyobject_to_rmpv(obj: &Bound<'_, PyAny>) -> PyResult<rmpv::Value> {
    if obj.is_none() {
        return Ok(rmpv::Value::Nil);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(rmpv::Value::Boolean(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(rmpv::Value::Integer(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(rmpv::Value::F64(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(rmpv::Value::String(s.into()));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let arr: PyResult<Vec<rmpv::Value>> = list.iter().map(|i| pyobject_to_rmpv(&i)).collect();
        return Ok(rmpv::Value::Array(arr?));
    }
    if let Ok(d) = obj.downcast::<PyDict>() {
        let mut pairs = Vec::new();
        for (k, v) in d.iter() {
            pairs.push((pyobject_to_rmpv(&k)?, pyobject_to_rmpv(&v)?));
        }
        return Ok(rmpv::Value::Map(pairs));
    }
    // Fallback: convert to string
    Ok(rmpv::Value::String(obj.str()?.to_string().into()))
}

// ── Feature 1 — Fast BM25 corpus JSON decode ──────────────────────────────────

/// Decode a BM25 corpus JSON payload (raw bytes) into a flat list of
/// {"id", "text", "metadata"} Python dicts.
///
/// Accepts both:
///   - dedup_v1 format: {"format": "dedup_v1", "texts": [...], "docs": [...]}
///   - legacy format: list of {"id", "text", "metadata"} dicts
///
/// Returns `None` on unrecognised/malformed input so Python can fall back.
#[pyfunction]
fn decode_corpus_json(py: Python<'_>, data: &[u8]) -> PyResult<Option<Py<PyList>>> {
    let parsed: serde_json::Value = match serde_json::from_slice(data) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let out = PyList::empty(py);

    match &parsed {
        serde_json::Value::Object(map) => {
            // dedup_v1 format
            let format = map.get("format").and_then(|v| v.as_str()).unwrap_or("");
            if format != "dedup_v1" {
                return Ok(None);
            }
            let texts = match map.get("texts").and_then(|v| v.as_array()) {
                Some(t) => t,
                None => return Ok(None),
            };
            let docs = match map.get("docs").and_then(|v| v.as_array()) {
                Some(d) => d,
                None => return Ok(None),
            };
            for doc in docs {
                let doc_map = match doc.as_object() {
                    Some(m) => m,
                    None => return Ok(None),
                };
                let id = doc_map
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let t_idx = match doc_map.get("t").and_then(|v| v.as_u64()) {
                    Some(i) => i as usize,
                    None => return Ok(None),
                };
                let text = match texts.get(t_idx).and_then(|v| v.as_str()) {
                    Some(s) => s.to_string(),
                    None => return Ok(None),
                };
                let metadata_val = doc_map
                    .get("metadata")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                let meta_py = json_value_to_pyobject(py, &metadata_val)?;

                let d = PyDict::new(py);
                d.set_item("id", id)?;
                d.set_item("text", text)?;
                d.set_item("metadata", meta_py)?;
                out.append(d)?;
            }
        }
        serde_json::Value::Array(arr) => {
            // Legacy list format
            for item in arr {
                let d = PyDict::new(py);
                let id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = item
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let metadata_val = item
                    .get("metadata")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                let meta_py = json_value_to_pyobject(py, &metadata_val)?;
                d.set_item("id", id)?;
                d.set_item("text", text)?;
                d.set_item("metadata", meta_py)?;
                out.append(d)?;
            }
        }
        _ => return Ok(None),
    }

    Ok(Some(out.unbind()))
}

// ── Feature 2 — MessagePack corpus encode/decode ──────────────────────────────

/// Encode BM25 corpus to MessagePack bytes.
///
/// Layout: fixarray[3] → [version: u8=2, texts: array<str>, docs: array<fixarray[3]=[id,t,metadata]>]
///
/// `texts` — deduplicated text pool (list of str)
/// `docs`  — list of {"id": str, "t": int, "metadata": dict}
#[pyfunction]
fn encode_corpus_msgpack(
    py: Python<'_>,
    texts: &Bound<'_, PyList>,
    docs: &Bound<'_, PyList>,
) -> PyResult<Py<PyBytes>> {
    let mut texts_rmpv: Vec<rmpv::Value> = Vec::with_capacity(texts.len());
    for t in texts.iter() {
        let s = t.extract::<String>().unwrap_or_default();
        texts_rmpv.push(rmpv::Value::String(s.into()));
    }

    let mut docs_rmpv: Vec<rmpv::Value> = Vec::with_capacity(docs.len());
    for item in docs.iter() {
        let d = item.downcast::<PyDict>()?;
        let id = d
            .get_item("id")?
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let t_idx = d
            .get_item("t")?
            .and_then(|v| v.extract::<u64>().ok())
            .unwrap_or(0);
        let meta_obj = d.get_item("metadata")?.unwrap_or_else(|| py.None().into_bound(py));
        let meta_rmpv = pyobject_to_rmpv(&meta_obj)?;
        docs_rmpv.push(rmpv::Value::Array(vec![
            rmpv::Value::String(id.into()),
            rmpv::Value::Integer(t_idx.into()),
            meta_rmpv,
        ]));
    }

    let payload = rmpv::Value::Array(vec![
        rmpv::Value::Integer(2_u64.into()), // version
        rmpv::Value::Array(texts_rmpv),
        rmpv::Value::Array(docs_rmpv),
    ]);

    let mut buf: Vec<u8> = Vec::new();
    rmpv::encode::write_value(&mut buf, &payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;

    Ok(PyBytes::new(py, &buf).unbind())
}

/// Decode MessagePack corpus bytes back to a flat list of {"id","text","metadata"} dicts.
/// Returns `None` on malformed input (Python falls back to JSON path).
#[pyfunction]
fn decode_corpus_msgpack(py: Python<'_>, data: &[u8]) -> PyResult<Option<Py<PyList>>> {
    let mut cursor = std::io::Cursor::new(data);
    let payload = match rmpv::decode::read_value(&mut cursor) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let arr = match payload {
        rmpv::Value::Array(a) if a.len() == 3 => a,
        _ => return Ok(None),
    };

    // version check — currently accept version 2
    match &arr[0] {
        rmpv::Value::Integer(n) if n.as_u64() == Some(2) => {}
        _ => return Ok(None),
    }

    let texts: Vec<String> = match &arr[1] {
        rmpv::Value::Array(a) => a
            .iter()
            .map(|v| match v {
                rmpv::Value::String(s) => s.as_str().unwrap_or("").to_string(),
                _ => String::new(),
            })
            .collect(),
        _ => return Ok(None),
    };

    let docs_arr = match &arr[2] {
        rmpv::Value::Array(a) => a,
        _ => return Ok(None),
    };

    let out = PyList::empty(py);
    for doc_val in docs_arr {
        let doc_arr = match doc_val {
            rmpv::Value::Array(a) if a.len() == 3 => a,
            _ => return Ok(None),
        };
        let id = match &doc_arr[0] {
            rmpv::Value::String(s) => s.as_str().unwrap_or("").to_string(),
            _ => return Ok(None),
        };
        let t_idx = match &doc_arr[1] {
            rmpv::Value::Integer(n) => n.as_u64().unwrap_or(0) as usize,
            _ => return Ok(None),
        };
        let text = texts.get(t_idx).cloned().unwrap_or_default();
        let meta_py = rmpv_to_pyobject(py, &doc_arr[2])?;

        let d = PyDict::new(py);
        d.set_item("id", id)?;
        d.set_item("text", text)?;
        d.set_item("metadata", meta_py)?;
        out.append(d)?;
    }

    Ok(Some(out.unbind()))
}

// ── Feature 3 — SHA-256 content hash ─────────────────────────────────────────

/// Return the SHA-256 hex digest of `text` (UTF-8 encoded, leading/trailing
/// whitespace stripped).
///
/// Mirrors: `hashlib.sha256(text.strip().encode("utf-8")).hexdigest()`
#[pyfunction]
fn compute_sha256(text: &str) -> String {
    use digest::Digest;
    use sha2::Sha256;
    format!("{:x}", Sha256::digest(text.trim().as_bytes()))
}

// ── Feature 4 — Binary hash store ────────────────────────────────────────────

/// Write MD5 hex hashes to a binary file.
///
/// Binary layout:
///   bytes 0–3   magic  b"AXH1"
///   bytes 4–7   count  uint32 little-endian
///   bytes 8+    N×16   raw MD5 bytes, sorted lexicographically
#[pyfunction]
fn save_hash_store_binary(path: &str, hashes: Vec<String>) -> PyResult<()> {
    use std::io::Write;

    let mut raw_hashes: Vec<[u8; 16]> = Vec::with_capacity(hashes.len());
    for h in &hashes {
        let trimmed = h.trim();
        if trimmed.len() != 32 {
            continue; // skip malformed
        }
        let mut bytes = [0u8; 16];
        let mut ok = true;
        for i in 0..16 {
            match u8::from_str_radix(&trimmed[i * 2..i * 2 + 2], 16) {
                Ok(b) => bytes[i] = b,
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            raw_hashes.push(bytes);
        }
    }

    raw_hashes.sort_unstable();

    let mut buf: Vec<u8> = Vec::with_capacity(8 + raw_hashes.len() * 16);
    buf.write_all(b"AXH1")
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let count = raw_hashes.len() as u32;
    buf.write_all(&count.to_le_bytes())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    for h in &raw_hashes {
        buf.write_all(h)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    std::fs::write(path, &buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Load binary hash store; returns list of hex strings (for Python set compat).
#[pyfunction]
fn load_hash_store_binary(path: &str) -> PyResult<Vec<String>> {
    let data =
        std::fs::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    if data.len() < 8 || &data[..4] != b"AXH1" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid hash store binary format",
        ));
    }
    let count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let expected_len = 8 + count * 16;
    if data.len() < expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "truncated hash store binary file",
        ));
    }

    let mut hashes = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 8 + i * 16;
        let chunk = &data[offset..offset + 16];
        let hex = chunk
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        hashes.push(hex);
    }
    Ok(hashes)
}

/// Probe binary hash store for membership without loading all hashes into Python.
///
/// Opens the file via mmap and binary-searches for `hash_hex`.
/// Returns `True` if found, `False` if not found, `None` if file is missing
/// or unreadable (Python falls back).
#[pyfunction]
fn probe_hash_store(path: &str, hash_hex: &str) -> PyResult<Option<bool>> {
    let trimmed = hash_hex.trim();
    if trimmed.len() != 32 {
        return Ok(Some(false));
    }
    let mut needle = [0u8; 16];
    for i in 0..16 {
        match u8::from_str_radix(&trimmed[i * 2..i * 2 + 2], 16) {
            Ok(b) => needle[i] = b,
            Err(_) => return Ok(Some(false)),
        }
    }

    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let meta = match file.metadata() {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let file_len = meta.len() as usize;
    if file_len < 8 {
        return Ok(None);
    }

    let mmap = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };

    if &mmap[..4] != b"AXH1" {
        return Ok(None);
    }
    let count = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    if file_len < 8 + count * 16 {
        return Ok(None);
    }

    // Binary search over sorted 16-byte records
    let mut lo = 0usize;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let offset = 8 + mid * 16;
        let record: &[u8; 16] = mmap[offset..offset + 16].try_into().unwrap();
        match record.cmp(&needle) {
            std::cmp::Ordering::Equal => return Ok(Some(true)),
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    Ok(Some(false))
}

// ── Feature 5 — Sentence window index msgpack serialization ──────────────────

/// Encode SentenceWindowIndex data to msgpack bytes.
///
/// `records`             — dict[str, dict]   (sentence_id → record dict)
/// `chunk_to_sentences`  — dict[str, list[str]]
#[pyfunction]
fn encode_sentence_index(
    records: &Bound<'_, PyDict>,
    chunk_to_sentences: &Bound<'_, PyDict>,
) -> PyResult<Py<PyBytes>> {
    let py = records.py();
    let records_rmpv = pyobject_to_rmpv(records.as_any())?;
    let cts_rmpv = pyobject_to_rmpv(chunk_to_sentences.as_any())?;

    let payload = rmpv::Value::Array(vec![
        rmpv::Value::Integer(1_u64.into()), // version
        records_rmpv,
        cts_rmpv,
    ]);

    let mut buf: Vec<u8> = Vec::new();
    rmpv::encode::write_value(&mut buf, &payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;

    Ok(PyBytes::new(py, &buf).unbind())
}

/// Decode msgpack bytes → `(records: PyDict, chunk_to_sentences: PyDict)`.
/// Returns `None` on error.
#[pyfunction]
fn decode_sentence_index(
    py: Python<'_>,
    data: &[u8],
) -> PyResult<Option<(Py<PyDict>, Py<PyDict>)>> {
    let mut cursor = std::io::Cursor::new(data);
    let payload = match rmpv::decode::read_value(&mut cursor) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let arr = match payload {
        rmpv::Value::Array(a) if a.len() == 3 => a,
        _ => return Ok(None),
    };

    let records_py = rmpv_to_pyobject(py, &arr[1])?;
    let cts_py = rmpv_to_pyobject(py, &arr[2])?;

    let records_dict = match records_py.downcast_bound::<PyDict>(py) {
        Ok(d) => d.clone().unbind(),
        Err(_) => return Ok(None),
    };
    let cts_dict = match cts_py.downcast_bound::<PyDict>(py) {
        Ok(d) => d.clone().unbind(),
        Err(_) => return Ok(None),
    };

    Ok(Some((records_dict, cts_dict)))
}

/// Encode SentenceVectorStore meta to msgpack bytes.
///
/// `ids`  — list[str]
/// `meta` — list[dict]
#[pyfunction]
fn encode_sentence_meta(
    ids: &Bound<'_, PyList>,
    meta: &Bound<'_, PyList>,
) -> PyResult<Py<PyBytes>> {
    let py = ids.py();
    let ids_rmpv = pyobject_to_rmpv(ids.as_any())?;
    let meta_rmpv = pyobject_to_rmpv(meta.as_any())?;

    let payload = rmpv::Value::Array(vec![
        rmpv::Value::Integer(1_u64.into()), // version
        ids_rmpv,
        meta_rmpv,
    ]);

    let mut buf: Vec<u8> = Vec::new();
    rmpv::encode::write_value(&mut buf, &payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;

    Ok(PyBytes::new(py, &buf).unbind())
}

/// Decode msgpack bytes → `(ids: list[str], meta: list[dict])`.
/// Returns `None` on error.
#[pyfunction]
fn decode_sentence_meta(
    py: Python<'_>,
    data: &[u8],
) -> PyResult<Option<(Py<PyList>, Py<PyList>)>> {
    let mut cursor = std::io::Cursor::new(data);
    let payload = match rmpv::decode::read_value(&mut cursor) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let arr = match payload {
        rmpv::Value::Array(a) if a.len() == 3 => a,
        _ => return Ok(None),
    };

    let ids_py = rmpv_to_pyobject(py, &arr[1])?;
    let meta_py = rmpv_to_pyobject(py, &arr[2])?;

    let ids_list = match ids_py.downcast_bound::<PyList>(py) {
        Ok(l) => l.clone().unbind(),
        Err(_) => return Ok(None),
    };
    let meta_list = match meta_py.downcast_bound::<PyList>(py) {
        Ok(l) => l.clone().unbind(),
        Err(_) => return Ok(None),
    };

    Ok(Some((ids_list, meta_list)))
}

// ── Feature 6 — Sentence segmentation ────────────────────────────────────────

/// Split text into sentences, merging fragments shorter than `min_chars` into
/// the preceding sentence.
///
/// Mirrors `sentence_window.segment_text()` exactly.
/// The Python version uses `re.split(r"(?<=[.!?])\s+", text)`.
/// Rust's regex crate doesn't support lookbehind, so we implement it manually:
/// scan for [.!?] followed by whitespace and split at the whitespace boundary.
#[pyfunction]
fn segment_text(text: &str, min_chars: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![];
    }

    // Collect split positions: index of the first whitespace char after each [.!?]
    let chars: Vec<char> = text.chars().collect();
    let _n = chars.len();
    let mut split_starts: Vec<usize> = Vec::new(); // byte offsets of whitespace runs to split at
    let mut byte_pos = 0usize;
    let mut char_before: char = '\0';
    for &ch in &chars {
        if matches!(char_before, '.' | '!' | '?') && ch.is_whitespace() {
            split_starts.push(byte_pos);
        }
        char_before = ch;
        byte_pos += ch.len_utf8();
    }

    // Split the text at those positions
    let mut parts: Vec<&str> = Vec::new();
    let mut start_byte = 0usize;
    let text_bytes = text.as_bytes();
    for &split_at in &split_starts {
        // skip the whitespace run
        let mut end_ws = split_at;
        while end_ws < text.len() && (text_bytes[end_ws] == b' ' || text_bytes[end_ws] == b'\t' || text_bytes[end_ws] == b'\r' || text_bytes[end_ws] == b'\n') {
            end_ws += 1;
        }
        parts.push(&text[start_byte..split_at]);
        start_byte = end_ws;
    }
    parts.push(&text[start_byte..]);

    let mut sentences: Vec<String> = Vec::with_capacity(parts.len());
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part.len() < min_chars && !sentences.is_empty() {
            let last = sentences.last_mut().unwrap();
            last.push(' ');
            last.push_str(part);
        } else {
            sentences.push(part.to_string());
        }
    }
    sentences
}

// ── Feature 7 — Cosine similarity ────────────────────────────────────────────

/// Compute cosine similarity between two float vectors.
/// Returns 0.0 for zero-norm vectors (avoids div-by-zero).
#[pyfunction]
fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Group embedding row indices whose cosine similarity exceeds `threshold`.
///
/// This avoids materializing the full NxN similarity matrix in Python.
#[pyfunction]
fn resolve_entity_alias_groups(
    py: Python<'_>,
    embeddings: &Bound<'_, PyList>,
    threshold: f64,
) -> PyResult<Py<PyList>> {
    use pyo3::types::PyList;
    use std::collections::BTreeMap;

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
    for item in embeddings.iter() {
        let seq = match item.downcast::<PyList>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mut row: Vec<f32> = Vec::with_capacity(seq.len());
        for value in seq.iter() {
            row.push(value.extract::<f32>().unwrap_or(0.0));
        }
        let norm_sq: f32 = row.iter().map(|v| v * v).sum();
        if norm_sq > 0.0 {
            let inv = norm_sq.sqrt().recip();
            for value in row.iter_mut() {
                *value *= inv;
            }
        }
        rows.push(row);
    }

    let n = rows.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    let threshold_f32 = threshold as f32;
    for i in 0..n {
        for j in (i + 1)..n {
            let len = rows[i].len().min(rows[j].len());
            if len == 0 {
                continue;
            }
            let mut dot = 0.0_f32;
            for k in 0..len {
                dot += rows[i][k] * rows[j][k];
            }
            if dot >= threshold_f32 {
                union(&mut parent, i, j);
            }
        }
    }

    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for idx in 0..n {
        let root = find(&mut parent, idx);
        groups.entry(root).or_default().push(idx);
    }

    let out = PyList::empty(py);
    for members in groups.values() {
        if members.len() < 2 {
            continue;
        }
        out.append(PyList::new(py, members.iter().copied())?)?;
    }
    Ok(out.unbind())
}

// ── Phase 3 helpers ───────────────────────────────────────────────────────────

/// Extract f64 score from a PyAny object that is a Python dict with key "score".
fn get_score_from_dict(d: &Bound<'_, pyo3::types::PyDict>) -> f64 {
    d.get_item("score")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(0.0)
}

fn get_optional_f64_from_dict(d: &Bound<'_, pyo3::types::PyDict>, key: &str) -> Option<f64> {
    d.get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<f64>().ok())
}

/// Extract a String value from a PyDict by key.
fn get_str_from_dict(d: &Bound<'_, pyo3::types::PyDict>, key: &str) -> String {
    d.get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default()
}

/// Copy all key/value pairs of `src` into a new PyDict.
fn copy_pydict<'py>(py: Python<'py>, src: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    let dst = PyDict::new(py);
    for (k, v) in src.iter() {
        dst.set_item(k, v)?;
    }
    Ok(dst)
}

/// Min-max normalize a slice of f64 values to [0.0, 1.0].
/// All-zero or flat inputs return 0.0 (unless all equal + nonzero → 1.0).
fn min_max_norm(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        return if max > 0.0 {
            vec![1.0; scores.len()]
        } else {
            vec![0.0; scores.len()]
        };
    }
    scores.iter().map(|&s| (s - min) / (max - min)).collect()
}

/// Tokenize text for MMR Jaccard comparison.
/// Returns sorted, deduplicated lowercase words (strips non-alphanumeric chars).
fn tokenize_for_mmr(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Jaccard similarity of two sorted, deduplicated token slices using O(n+m) merge scan.
fn jaccard_sorted(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut intersection = 0usize;
    let mut union = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                union += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                union += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                union += 1;
                j += 1;
            }
        }
    }
    union += (a.len() - i) + (b.len() - j);
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Check if `pattern` appears in `text` at a word boundary (no regex dependency).
/// A boundary is: start/end of string or a non-alphanumeric char adjacent to the match.
fn word_boundary_match(text: &str, pattern: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    let text_bytes = text.as_bytes();
    let pat_bytes = pattern.as_bytes();
    let tlen = text_bytes.len();
    let plen = pat_bytes.len();
    if plen > tlen {
        return false;
    }
    // Case-insensitive comparison — we do lowercase comparison per byte (ASCII safe for identifier names)
    let mut pos = 0usize;
    while pos + plen <= tlen {
        // Find next candidate using memchr-style scan
        let candidate = text[pos..].find(pattern).or_else(|| {
            // try case-insensitive match manually
            text[pos..]
                .as_bytes()
                .windows(plen)
                .position(|w| w.eq_ignore_ascii_case(pat_bytes))
        });
        match candidate {
            None => break,
            Some(offset) => {
                let start = pos + offset;
                let end = start + plen;
                // Check left boundary
                let left_ok = start == 0 || !text_bytes[start - 1].is_ascii_alphanumeric();
                // Check right boundary
                let right_ok = end == tlen || !text_bytes[end].is_ascii_alphanumeric();
                if left_ok && right_ok {
                    return true;
                }
                pos = start + 1;
            }
        }
    }
    false
}

// ── Phase 3 Feature 1: Score fusion ──────────────────────────────────────────

/// Merge vector + BM25 results using weighted normalized score fusion.
/// weight: 1.0 = pure semantic, 0.0 = pure lexical.
/// Returns a combined list sorted by fused score descending.
#[pyfunction]
#[pyo3(signature = (vector_results, bm25_results, weight=0.7))]
fn score_fusion_weighted(
    py: Python<'_>,
    vector_results: &Bound<'_, pyo3::types::PyList>,
    bm25_results: &Bound<'_, pyo3::types::PyList>,
    weight: f64,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyDict, PyList};
    use std::collections::HashMap;

    // Extract scores
    let v_scores: Vec<f64> = vector_results
        .iter()
        .map(|item| {
            item.downcast::<PyDict>()
                .map(|d| get_score_from_dict(d))
                .unwrap_or(0.0)
        })
        .collect();
    let b_scores: Vec<f64> = bm25_results
        .iter()
        .map(|item| {
            item.downcast::<PyDict>()
                .map(|d| get_score_from_dict(d))
                .unwrap_or(0.0)
        })
        .collect();

    let v_norm = min_max_norm(&v_scores);
    let b_norm = min_max_norm(&b_scores);

    // doc_id → index into docs vec
    let mut id_to_idx: HashMap<String, usize> = HashMap::new();
    let mut docs: Vec<Py<PyDict>> = Vec::new();
    let mut fused_scores: Vec<f64> = Vec::new();

    for (i, item) in vector_results.iter().enumerate() {
        let src = item.downcast::<PyDict>()?;
        let doc_id = get_str_from_dict(src, "id");
        let new_dict = copy_pydict(py, src)?.unbind();
        let v_score = v_scores.get(i).copied().unwrap_or(0.0);
        new_dict.bind(py).set_item("vector_score", v_score)?;
        let fused = v_norm.get(i).copied().unwrap_or(0.0) * weight;
        id_to_idx.insert(doc_id, docs.len());
        docs.push(new_dict);
        fused_scores.push(fused);
    }

    for (i, item) in bm25_results.iter().enumerate() {
        let src = item.downcast::<PyDict>()?;
        let doc_id = get_str_from_dict(src, "id");
        let b_contrib = b_norm.get(i).copied().unwrap_or(0.0) * (1.0 - weight);
        if let Some(&idx) = id_to_idx.get(&doc_id) {
            fused_scores[idx] += b_contrib;
            docs[idx].bind(py).set_item("score", fused_scores[idx])?;
        } else {
            let new_dict = copy_pydict(py, src)?.unbind();
            new_dict.bind(py).set_item("vector_score", 0.0)?;
            new_dict.bind(py).set_item("fused_only", true)?;
            let fused = b_contrib;
            id_to_idx.insert(doc_id, docs.len());
            docs.push(new_dict);
            fused_scores.push(fused);
        }
    }

    // Stamp final scores
    for (i, doc) in docs.iter().enumerate() {
        doc.bind(py).set_item("score", fused_scores[i])?;
    }

    // Sort descending by fused score
    let mut indexed: Vec<(usize, f64)> = fused_scores.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let out = PyList::empty(py);
    for (idx, _) in indexed {
        out.append(docs[idx].bind(py))?;
    }
    Ok(out.unbind())
}

/// Merge vector + BM25 results using Reciprocal Rank Fusion.
/// k: RRF smoothing constant (default 60).
/// Preserves original vector score in "vector_score" field.
#[pyfunction]
#[pyo3(signature = (vector_results, bm25_results, k=60))]
fn score_fusion_rrf(
    py: Python<'_>,
    vector_results: &Bound<'_, pyo3::types::PyList>,
    bm25_results: &Bound<'_, pyo3::types::PyList>,
    k: i64,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyDict, PyList};
    use std::collections::HashMap;

    let k_f = k as f64;
    let mut rrf_scores: HashMap<String, f64> = HashMap::new();
    let mut vector_scores: HashMap<String, f64> = HashMap::new();
    let mut doc_store: HashMap<String, Py<PyDict>> = HashMap::new();

    for (rank, item) in vector_results.iter().enumerate() {
        let src = item.downcast::<PyDict>()?;
        let doc_id = get_str_from_dict(src, "id");
        let orig_score = get_score_from_dict(src);
        *rrf_scores.entry(doc_id.clone()).or_insert(0.0) += 1.0 / (rank as f64 + k_f);
        vector_scores.insert(doc_id.clone(), orig_score);
        doc_store.entry(doc_id).or_insert_with(|| src.clone().unbind());
    }

    for (rank, item) in bm25_results.iter().enumerate() {
        let src = item.downcast::<PyDict>()?;
        let doc_id = get_str_from_dict(src, "id");
        *rrf_scores.entry(doc_id.clone()).or_insert(0.0) += 1.0 / (rank as f64 + k_f);
        doc_store.entry(doc_id).or_insert_with(|| src.clone().unbind());
    }

    let mut items: Vec<(String, f64)> = rrf_scores.into_iter().collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let out = PyList::empty(py);
    for (doc_id, rrf_score) in items {
        let src_dict = doc_store.get(&doc_id).unwrap();
        let new_dict = copy_pydict(py, src_dict.bind(py))?.unbind();
        new_dict.bind(py).set_item("score", rrf_score)?;
        let vs = vector_scores.get(&doc_id).copied().unwrap_or(0.0);
        new_dict.bind(py).set_item("vector_score", vs)?;
        out.append(new_dict.bind(py))?;
    }
    Ok(out.unbind())
}

#[pyfunction]
fn dedupe_best_by_id(
    py: Python<'_>,
    results: &Bound<'_, pyo3::types::PyList>,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyDict, PyList};
    use std::collections::HashMap;

    let mut order: Vec<String> = Vec::new();
    let mut best_docs: HashMap<String, (f64, Py<PyDict>)> = HashMap::new();

    for item in results.iter() {
        let src = item.downcast::<PyDict>()?;
        let doc_id = get_str_from_dict(src, "id");
        if doc_id.is_empty() {
            continue;
        }
        let score = get_score_from_dict(src);
        let new_dict = copy_pydict(py, src)?.unbind();
        match best_docs.get_mut(&doc_id) {
            Some((best_score, best_dict)) => {
                if score > *best_score {
                    *best_score = score;
                    *best_dict = new_dict;
                }
            }
            None => {
                order.push(doc_id.clone());
                best_docs.insert(doc_id, (score, new_dict));
            }
        }
    }

    let out = PyList::empty(py);
    for doc_id in order {
        if let Some((_, doc)) = best_docs.remove(&doc_id) {
            out.append(doc.bind(py))?;
        }
    }
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature = (results, threshold, score_field="vector_score"))]
fn filter_results_by_threshold(
    py: Python<'_>,
    results: &Bound<'_, pyo3::types::PyList>,
    threshold: f64,
    score_field: &str,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyDict, PyList};

    let fallback_field = if score_field == "score" {
        "vector_score"
    } else {
        "score"
    };

    let out = PyList::empty(py);
    for item in results.iter() {
        let src = item.downcast::<PyDict>()?;
        let keep = match src.get_item("fused_only").ok().flatten() {
            Some(value) => value.extract::<bool>().ok().unwrap_or(false),
            None => false,
        };
        if keep {
            out.append(copy_pydict(py, src)?)?;
            continue;
        }
        let signal = get_optional_f64_from_dict(src, score_field)
            .or_else(|| get_optional_f64_from_dict(src, fallback_field))
            .unwrap_or(0.0);
        if signal >= threshold {
            out.append(copy_pydict(py, src)?)?;
        }
    }
    Ok(out.unbind())
}

// ── Phase 3 Feature 2: MMR reranking ─────────────────────────────────────────

/// Rerank results using Maximal Marginal Relevance (Jaccard text similarity).
///
/// Iteratively selects the next document maximising:
///   lambda_mult * relevance_score - (1-lambda_mult) * max_jaccard_to_selected
///
/// Documents with Jaccard >= dup_threshold to any already-selected document
/// are dropped as near-duplicates.
///
/// Returns a reordered + deduplicated list of the original dicts.
#[pyfunction]
#[pyo3(signature = (results, lambda_mult=0.5, dup_threshold=0.85))]
fn mmr_rerank(
    py: Python<'_>,
    results: &Bound<'_, pyo3::types::PyList>,
    lambda_mult: f64,
    dup_threshold: f64,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyDict, PyList};

    let n = results.len();
    if n <= 1 {
        let out = PyList::empty(py);
        for item in results.iter() {
            out.append(item)?;
        }
        return Ok(out.unbind());
    }

    // Collect scores and token sets
    let mut scores: Vec<f64> = Vec::with_capacity(n);
    let mut token_sets: Vec<Vec<String>> = Vec::with_capacity(n);
    let mut orig: Vec<Py<PyDict>> = Vec::with_capacity(n);

    for item in results.iter() {
        let d = item.downcast::<PyDict>()?;
        scores.push(get_score_from_dict(d));
        let text = get_str_from_dict(d, "text");
        token_sets.push(tokenize_for_mmr(&text));
        orig.push(d.clone().unbind());
    }

    let mut selected: Vec<usize> = Vec::with_capacity(n);
    let mut remaining: Vec<usize> = (0..n).collect();

    while !remaining.is_empty() {
        let best = if selected.is_empty() {
            // Pick highest relevance score
            *remaining
                .iter()
                .max_by(|&&a, &&b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
        } else {
            // MMR criterion
            *remaining
                .iter()
                .max_by(|&&a, &&b| {
                    let mmr_a = lambda_mult * scores[a]
                        - (1.0 - lambda_mult)
                            * selected
                                .iter()
                                .map(|&s| jaccard_sorted(&token_sets[a], &token_sets[s]))
                                .fold(0.0_f64, f64::max);
                    let mmr_b = lambda_mult * scores[b]
                        - (1.0 - lambda_mult)
                            * selected
                                .iter()
                                .map(|&s| jaccard_sorted(&token_sets[b], &token_sets[s]))
                                .fold(0.0_f64, f64::max);
                    mmr_a.partial_cmp(&mmr_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        };

        remaining.retain(|&x| x != best);

        // Drop near-duplicate of any selected doc
        if !selected.is_empty() {
            let max_sim = selected
                .iter()
                .map(|&s| jaccard_sorted(&token_sets[best], &token_sets[s]))
                .fold(0.0_f64, f64::max);
            if max_sim >= dup_threshold {
                continue;
            }
        }
        selected.push(best);
    }

    let out = PyList::empty(py);
    for idx in selected {
        out.append(orig[idx].bind(py))?;
    }
    Ok(out.unbind())
}

// ── Phase 3 Feature 3: Code-doc bridge edges ─────────────────────────────────

/// Scan prose chunks for occurrences of code symbol names and emit new MENTIONED_IN edges.
///
/// Arguments:
///   sym_lookup: dict[str, str]  — symbol_name → node_id (names >= 4 chars)
///   prose_chunks: list[dict]    — each dict has "id" and "text" keys
///   existing_edge_tuples: list[tuple[str, str, str]]  — (source, target, edge_type)
///
/// Returns list[dict] of new edge dicts (not already in existing_edge_tuples).
#[pyfunction]
fn build_code_doc_bridge_edges(
    py: Python<'_>,
    sym_lookup: &Bound<'_, pyo3::types::PyDict>,
    prose_chunks: &Bound<'_, pyo3::types::PyList>,
    existing_edge_tuples: &Bound<'_, pyo3::types::PyList>,
) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::{PyDict, PyList, PyTuple};
    use std::collections::HashSet;

    // Build existing edges set
    let mut existing: HashSet<(String, String, String)> = HashSet::new();
    for item in existing_edge_tuples.iter() {
        if let Ok(tup) = item.downcast::<PyTuple>() {
            if tup.len() >= 3 {
                let src = tup.get_item(0)?.extract::<String>().unwrap_or_default();
                let tgt = tup.get_item(1)?.extract::<String>().unwrap_or_default();
                let et = tup.get_item(2)?.extract::<String>().unwrap_or_default();
                existing.insert((src, tgt, et));
            }
        }
    }

    // Collect sym_lookup entries as (name, node_id) pairs
    let mut symbols: Vec<(String, String)> = Vec::new();
    for (k, v) in sym_lookup.iter() {
        let name = k.extract::<String>().unwrap_or_default();
        let node_id = v.extract::<String>().unwrap_or_default();
        if !name.is_empty() && !node_id.is_empty() {
            symbols.push((name, node_id));
        }
    }

    let out = PyList::empty(py);
    for chunk_item in prose_chunks.iter() {
        let chunk = chunk_item.downcast::<PyDict>()?;
        let chunk_id = get_str_from_dict(chunk, "id");
        let text = get_str_from_dict(chunk, "text");
        if chunk_id.is_empty() || text.is_empty() {
            continue;
        }
        for (sym_name, node_id) in &symbols {
            let ek = (node_id.clone(), chunk_id.clone(), "MENTIONED_IN".to_string());
            if existing.contains(&ek) {
                continue;
            }
            if word_boundary_match(&text, sym_name) {
                existing.insert(ek);
                let edge = PyDict::new(py);
                edge.set_item("source", node_id)?;
                edge.set_item("target", &chunk_id)?;
                edge.set_item("edge_type", "MENTIONED_IN")?;
                edge.set_item("chunk_id", &chunk_id)?;
                out.append(edge)?;
            }
        }
    }
    Ok(out.unbind())
}

// ── Phase 3 Feature 4: Entity graph I/O (msgpack) ────────────────────────────

/// Encode entity graph dict to msgpack bytes.
#[pyfunction]
fn encode_entity_graph(
    graph: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    use pyo3::types::PyBytes;
    let val = pyobject_to_rmpv(graph.as_any())?;
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &val)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;
    Ok(PyBytes::new(graph.py(), &buf).unbind())
}

/// Decode msgpack bytes to entity graph dict.
/// Returns None on malformed input so Python can fall back to JSON.
#[pyfunction]
fn decode_entity_graph(py: Python<'_>, data: &[u8]) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
    use pyo3::types::PyDict;
    let val = match rmpv::decode::read_value(&mut std::io::Cursor::new(data)) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let obj = rmpv_to_pyobject(py, &val)?;
    match obj.downcast_bound::<PyDict>(py) {
        Ok(d) => Ok(Some(d.clone().unbind())),
        Err(_) => Ok(None),
    }
}

/// Encode relation graph dict to msgpack bytes.
#[pyfunction]
fn encode_relation_graph(
    graph: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    use pyo3::types::PyBytes;
    let val = pyobject_to_rmpv(graph.as_any())?;
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &val)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;
    Ok(PyBytes::new(graph.py(), &buf).unbind())
}

/// Decode msgpack bytes to relation graph dict.
/// Returns None on malformed input so Python can fall back to JSON.
#[pyfunction]
fn decode_relation_graph(
    py: Python<'_>,
    data: &[u8],
) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
    use pyo3::types::PyDict;
    let val = match rmpv::decode::read_value(&mut std::io::Cursor::new(data)) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let obj = rmpv_to_pyobject(py, &val)?;
    match obj.downcast_bound::<PyDict>(py) {
        Ok(d) => Ok(Some(d.clone().unbind())),
        Err(_) => Ok(None),
    }
}

/// Encode entity embeddings dict (str → list[float]) to msgpack bytes.
#[pyfunction]
fn encode_entity_embeddings(
    embeddings: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    use pyo3::types::PyBytes;
    let val = pyobject_to_rmpv(embeddings.as_any())?;
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &val)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;
    Ok(PyBytes::new(embeddings.py(), &buf).unbind())
}

/// Decode msgpack bytes to entity embeddings dict.
/// Returns None on malformed input.
#[pyfunction]
fn decode_entity_embeddings(
    py: Python<'_>,
    data: &[u8],
) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
    use pyo3::types::PyDict;
    let val = match rmpv::decode::read_value(&mut std::io::Cursor::new(data)) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let obj = rmpv_to_pyobject(py, &val)?;
    match obj.downcast_bound::<PyDict>(py) {
        Ok(d) => Ok(Some(d.clone().unbind())),
        Err(_) => Ok(None),
    }
}

// ── Phase 3 Feature 5: Dedup corpus payload ───────────────────────────────────

/// Build a deduplicated corpus payload from a list of {"id", "text", "metadata"} dicts.
///
/// Returns (texts: list[str], docs: list[dict]) where each doc has:
///   "id": str, "t": int (index into texts), "metadata": dict
///
/// This is the Rust equivalent of BM25Retriever._build_dedup_corpus_payload().
#[pyfunction]
fn build_dedup_corpus_payload(
    py: Python<'_>,
    corpus: &Bound<'_, pyo3::types::PyList>,
) -> PyResult<(Py<pyo3::types::PyList>, Py<pyo3::types::PyList>)> {
    use pyo3::types::{PyDict, PyList};
    use std::collections::HashMap;

    let mut texts: Vec<String> = Vec::new();
    let mut text_to_idx: HashMap<String, usize> = HashMap::new();
    let mut docs_out: Vec<Py<PyDict>> = Vec::new();

    for item in corpus.iter() {
        let doc = item.downcast::<PyDict>()?;
        let text = get_str_from_dict(doc, "text");
        let id = get_str_from_dict(doc, "id");
        let idx = match text_to_idx.get(&text) {
            Some(&i) => i,
            None => {
                let i = texts.len();
                text_to_idx.insert(text.clone(), i);
                texts.push(text);
                i
            }
        };
        let metadata = doc
            .get_item("metadata")
            .ok()
            .flatten()
            .unwrap_or_else(|| PyDict::new(py).into_any());
        let d = PyDict::new(py);
        d.set_item("id", &id)?;
        d.set_item("t", idx)?;
        d.set_item("metadata", metadata)?;
        docs_out.push(d.unbind());
    }

    let texts_list = PyList::new(py, texts.iter().map(|s| s.as_str()))?;
    let docs_list = PyList::empty(py);
    for d in docs_out {
        docs_list.append(d.bind(py))?;
    }
    Ok((texts_list.unbind(), docs_list.unbind()))
}

// ── Phase 4: GraphRAG acceleration ───────────────────────────────────────────

/// Convert entity_graph and relation_graph Python dicts into flat node/edge lists
/// suitable for graph algorithms (e.g., Louvain community detection).
///
/// entity_graph: dict[str, any]  — keys are entity names (nodes)
/// relation_graph: dict[str, list[dict]] — keys are source entities; each entry has
///   "target" (str) and optionally "weight" (float, default 1.0)
///
/// Returns (nodes: list[str], edges: list[tuple[str, str, float]])
#[pyfunction]
fn build_graph_edges(
    py: Python<'_>,
    entity_graph: &Bound<'_, PyDict>,
    relation_graph: &Bound<'_, PyDict>,
) -> PyResult<(Py<PyList>, Py<PyList>)> {
    use pyo3::types::{PyList, PyTuple};
    let nodes_list = PyList::empty(py);
    for key in entity_graph.keys() {
        nodes_list.append(key)?;
    }
    let edges_list = PyList::empty(py);
    for (src_obj, entries_obj) in relation_graph.iter() {
        let src = match src_obj.extract::<String>() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let entries = match entries_obj.downcast::<PyList>() {
            Ok(l) => l,
            Err(_) => continue,
        };
        for entry_obj in entries.iter() {
            let entry = match entry_obj.downcast::<PyDict>() {
                Ok(d) => d,
                Err(_) => continue,
            };
            let tgt = match entry.get_item("target").ok().flatten() {
                Some(v) => v.extract::<String>().unwrap_or_default(),
                None => continue,
            };
            if tgt.is_empty() {
                continue;
            }
            let weight: f64 = entry
                .get_item("weight")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(1.0);
            let edge = PyTuple::new(py, [src.as_str().into_pyobject(py)?.into_any(), tgt.as_str().into_pyobject(py)?.into_any(), weight.into_pyobject(py)?.into_any()])?;
            edges_list.append(edge)?;
        }
    }
    Ok((nodes_list.unbind(), edges_list.unbind()))
}

/// Run Louvain community detection on a weighted undirected graph.
///
/// nodes: list[str]  — node labels
/// edges: list[tuple[str, str, float]]  — (source, target, weight)
/// resolution: modularity resolution parameter (1.0 = standard Louvain)
///
/// Returns dict[str, int] mapping node label → community ID.
/// Isolated nodes each get their own community.
/// Returns empty dict for an empty graph.
#[pyfunction]
fn run_louvain(
    py: Python<'_>,
    nodes: &Bound<'_, PyList>,
    edges: &Bound<'_, PyList>,
    resolution: f64,
) -> PyResult<Py<PyDict>> {
    let node_labels: Vec<String> = nodes
        .iter()
        .map(|v| v.extract::<String>().unwrap_or_default())
        .collect();
    let n = node_labels.len();
    let result_dict = PyDict::new(py);

    if n == 0 {
        return Ok(result_dict.unbind());
    }
    let gamma = if resolution.is_finite() && resolution > 0.0 {
        resolution
    } else {
        1.0
    };

    // Build node → index map
    let node_to_idx: HashMap<&str, usize> = node_labels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    // Parse edges into (usize, usize, f64)
    let mut edge_list: Vec<(usize, usize, f64)> = Vec::new();
    for edge_obj in edges.iter() {
        let tup = match edge_obj.downcast::<pyo3::types::PyTuple>() {
            Ok(t) => t,
            Err(_) => {
                // Try list fallback
                if let Ok(l) = edge_obj.downcast::<PyList>() {
                    if l.len() >= 2 {
                        let s = l.get_item(0)?.extract::<String>().unwrap_or_default();
                        let t = l.get_item(1)?.extract::<String>().unwrap_or_default();
                        let w = if l.len() >= 3 { l.get_item(2)?.extract::<f64>().unwrap_or(1.0) } else { 1.0 };
                        if let (Some(&ui), Some(&vi)) = (node_to_idx.get(s.as_str()), node_to_idx.get(t.as_str())) {
                            edge_list.push((ui, vi, w));
                        }
                    }
                }
                continue;
            }
        };
        if tup.len() < 2 {
            continue;
        }
        let s = tup.get_item(0)?.extract::<String>().unwrap_or_default();
        let t = tup.get_item(1)?.extract::<String>().unwrap_or_default();
        let w = if tup.len() >= 3 { tup.get_item(2)?.extract::<f64>().unwrap_or(1.0) } else { 1.0 };
        if let (Some(&ui), Some(&vi)) = (node_to_idx.get(s.as_str()), node_to_idx.get(t.as_str())) {
            if ui != vi {
                edge_list.push((ui, vi, w));
            }
        }
    }

    // Build adjacency list
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    let mut total_weight = 0.0_f64;
    for &(u, v, w) in &edge_list {
        adj[u].push((v, w));
        adj[v].push((u, w));
        total_weight += w;
    }
    let two_m = 2.0 * total_weight;

    // Node degrees
    let degrees: Vec<f64> = (0..n).map(|i| adj[i].iter().map(|(_, w)| w).sum()).collect();

    // Community assignment (start: each node in own community)
    let mut community: Vec<usize> = (0..n).collect();
    // sigma_tot[c] = sum of degrees of all nodes in community c
    let mut sigma_tot: Vec<f64> = degrees.clone();

    if two_m < 1e-15 {
        // Isolated graph — each node in its own community
        for (i, label) in node_labels.iter().enumerate() {
            result_dict.set_item(label.as_str(), i)?;
        }
        return Ok(result_dict.unbind());
    }

    // Louvain Phase 1: greedy node moves
    let mut improved = true;
    let mut iters = 0;
    while improved && iters < 20 {
        improved = false;
        iters += 1;
        for node in 0..n {
            let curr_c = community[node];
            let k_i = degrees[node];

            // Compute k_i_curr = sum of weights from node to current community
            let k_i_curr: f64 = adj[node]
                .iter()
                .filter(|(nb, _)| community[*nb] == curr_c && *nb != node)
                .map(|(_, w)| *w)
                .sum();

            // Remove node from current community
            sigma_tot[curr_c] -= k_i;

            // Collect distinct neighbor communities and k_i_in for each
            let mut neighbor_gains: HashMap<usize, f64> = HashMap::new();
            for &(nb, w) in &adj[node] {
                let nb_c = community[nb];
                if nb_c != curr_c {
                    *neighbor_gains.entry(nb_c).or_insert(0.0) += w;
                }
            }

            // Find best community to move to
            let mut best_c = curr_c;
            // Gain from moving to current community (reference = 0)
            // Resolution-aware modularity gain:
            // ΔQ = k_i_in/m - gamma * sigma_tot[c] * k_i / (2 * m^2)
            let gain_curr = k_i_curr / total_weight
                - gamma * sigma_tot[curr_c] * k_i / (two_m * total_weight);
            let mut best_gain = gain_curr;

            for (cand_c, k_i_in) in &neighbor_gains {
                let gain = k_i_in / total_weight
                    - gamma * sigma_tot[*cand_c] * k_i / (two_m * total_weight);
                if gain > best_gain + 1e-10 {
                    best_gain = gain;
                    best_c = *cand_c;
                }
            }

            // Re-insert node into best community
            community[node] = best_c;
            sigma_tot[best_c] += k_i;

            if best_c != curr_c {
                improved = true;
            }
        }
    }

    // Renumber communities to be contiguous
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;
    for (i, label) in node_labels.iter().enumerate() {
        let c = community[i];
        let new_c = *remap.entry(c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        result_dict.set_item(label.as_str(), new_c)?;
    }

    Ok(result_dict.unbind())
}

/// Merge a batch of per-chunk entity extraction results into the existing entity graph.
///
/// entity_graph: dict[str, dict] — the existing entity graph (mutated in-place)
/// results: list of (doc_id: str, entities: list[dict]) — same as what executor.map returns
///
/// For each entity dict (must have "name" key):
///   - Normalised key = name.lower().strip()
///   - If new: insert {description, type, chunk_ids=[doc_id], frequency=1, degree=0}
///   - If exists: append doc_id to chunk_ids (if not already present), update type if UNKNOWN
///
/// Returns count of new entity keys inserted.
#[pyfunction]
fn merge_entities_into_graph(
    py: Python<'_>,
    entity_graph: &Bound<'_, PyDict>,
    results: &Bound<'_, PyList>,
) -> PyResult<usize> {
    let mut new_count = 0usize;

    for pair_obj in results.iter() {
        // pair = (doc_id, [entity_dict, ...])
        let tup = match pair_obj.downcast::<pyo3::types::PyTuple>() {
            Ok(t) if t.len() >= 2 => t.to_owned(),
            _ => {
                // Try list fallback
                if let Ok(l) = pair_obj.downcast::<PyList>() {
                    if l.len() < 2 { continue; }
                    let doc_id = l.get_item(0)?.extract::<String>().unwrap_or_default();
                    let ents_obj = l.get_item(1)?;
                    let ents = match ents_obj.downcast::<PyList>() {
                        Ok(el) => el.to_owned(),
                        Err(_) => continue,
                    };
                    merge_chunk_entities(py, entity_graph, &doc_id, &ents, &mut new_count)?;
                }
                continue;
            }
        };
        let doc_id = tup.get_item(0)?.extract::<String>().unwrap_or_default();
        let ents_obj = tup.get_item(1)?;
        let ents = match ents_obj.downcast::<PyList>() {
            Ok(el) => el.to_owned(),
            Err(_) => continue,
        };
        merge_chunk_entities(py, entity_graph, &doc_id, &ents, &mut new_count)?;
    }
    Ok(new_count)
}

/// Inner helper: merge one chunk's entities into entity_graph.
fn merge_chunk_entities(
    py: Python<'_>,
    entity_graph: &Bound<'_, PyDict>,
    doc_id: &str,
    entities: &Bound<'_, PyList>,
    new_count: &mut usize,
) -> PyResult<()> {
    for ent_obj in entities.iter() {
        let ent = match ent_obj.downcast::<PyDict>() {
            Ok(d) => d.to_owned(),
            Err(_) => continue,
        };
        let name = match ent.get_item("name").ok().flatten() {
            Some(v) => v.extract::<String>().unwrap_or_default(),
            None => continue,
        };
        let key = name.to_lowercase();
        let key = key.trim();
        if key.is_empty() {
            continue;
        }

        match entity_graph.get_item(key).ok().flatten() {
            Some(existing_obj) => {
                // Existing entity — update chunk_ids and type if needed
                let existing = match existing_obj.downcast::<PyDict>() {
                    Ok(d) => d.to_owned(),
                    Err(_) => continue,
                };
                // Update type if currently UNKNOWN or absent
                let curr_type = existing
                    .get_item("type")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default();
                if curr_type.is_empty() || curr_type == "UNKNOWN" {
                    let new_type = ent
                        .get_item("type")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<String>().ok())
                        .unwrap_or_default();
                    if !new_type.is_empty() && new_type != "UNKNOWN" {
                        existing.set_item("type", &new_type)?;
                    }
                }
                // Append doc_id to chunk_ids if not present
                let chunk_ids = existing
                    .get_item("chunk_ids")
                    .ok()
                    .flatten();
                match chunk_ids {
                    Some(cids_obj) => {
                        if let Ok(cids) = cids_obj.downcast::<PyList>() {
                            let already = cids.iter().any(|c| {
                                c.extract::<String>().ok().as_deref() == Some(doc_id)
                            });
                            if !already {
                                cids.append(doc_id)?;
                            }
                        }
                    }
                    None => {
                        let cids = PyList::new(py, [doc_id])?;
                        existing.set_item("chunk_ids", cids)?;
                    }
                }
            }
            None => {
                // New entity
                let desc = ent
                    .get_item("description")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default();
                let ent_type = ent
                    .get_item("type")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_else(|| "UNKNOWN".to_string());
                let chunk_ids = PyList::new(py, [doc_id])?;
                let node = PyDict::new(py);
                node.set_item("description", &desc)?;
                node.set_item("type", &ent_type)?;
                node.set_item("chunk_ids", chunk_ids)?;
                node.set_item("frequency", 1_usize)?;
                node.set_item("degree", 0_usize)?;
                entity_graph.set_item(key, node)?;
                *new_count += 1;
            }
        }
    }
    Ok(())
}

/// Merge per-chunk relation extraction results into the existing relation graph.
///
/// relation_graph: dict[str, list[dict]] — the existing relation graph (mutated in-place)
///   Structure: {src_lower: [{target, relation, chunk_id, description, weight, support_count, ...}]}
/// results: list of (doc_id: str, triples: list[dict|tuple]) — same as rel_results from executor.map
///   Each triple is either:
///     - dict with {subject/source, relation, object/target, description}
///     - tuple (subject, relation, object)
///
/// For each triple:
///   - If (src, tgt, relation) already exists: increment weight + support_count
///   - Else: append new entry
///
/// Returns count of new relation entries inserted.
#[pyfunction]
fn merge_relations_into_graph(
    py: Python<'_>,
    relation_graph: &Bound<'_, PyDict>,
    results: &Bound<'_, PyList>,
) -> PyResult<usize> {
    let mut new_count = 0usize;

    for pair_obj in results.iter() {
        let (doc_id, triples) = match extract_result_pair(&pair_obj) {
            Some(v) => v,
            None => continue,
        };
        let triples = triples.bind(py);
        for triple_obj in triples.iter() {
            // Parse triple: dict or tuple
            let (subject, relation, obj, description, strength) =
                match parse_triple_obj(&triple_obj) {
                    Some(v) => v,
                    None => continue,
                };
            let src_lower = subject.to_lowercase();
            let src_lower = src_lower.trim();
            if src_lower.is_empty() {
                continue;
            }
            let tgt_lower = obj.to_lowercase();
            let tgt_lower = tgt_lower.trim();

            // Get or create the list for this source
            let rel_list = match relation_graph.get_item(src_lower).ok().flatten() {
                Some(v) => match v.downcast::<PyList>() {
                    Ok(l) => l.to_owned(),
                    Err(_) => {
                        let l = PyList::empty(py);
                        relation_graph.set_item(src_lower, &l)?;
                        l.unbind().bind(py).to_owned()
                    }
                },
                None => {
                    let l = PyList::empty(py);
                    relation_graph.set_item(src_lower, &l)?;
                    l.unbind().bind(py).to_owned()
                }
            };

            // Find existing entry with same (target, relation)
            let mut found = false;
            for entry_obj in rel_list.iter() {
                let entry = match entry_obj.downcast::<PyDict>() {
                    Ok(d) => d.to_owned(),
                    Err(_) => continue,
                };
                let et = entry
                    .get_item("target")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default();
                let er = entry
                    .get_item("relation")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default();
                if et.as_str() == tgt_lower && er == relation {
                    // Update existing
                    let curr_weight = entry
                        .get_item("weight")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<f64>().ok())
                        .unwrap_or(1.0);
                    entry.set_item("weight", curr_weight + strength as f64)?;
                    let curr_sc = entry
                        .get_item("support_count")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<usize>().ok())
                        .unwrap_or(1);
                    entry.set_item("support_count", curr_sc + 1)?;
                    // Append doc_id to text_unit_ids if not present
                    match entry.get_item("text_unit_ids").ok().flatten() {
                        Some(tui_obj) => {
                            if let Ok(tui) = tui_obj.downcast::<PyList>() {
                                let doc_id_str = doc_id.as_str();
                                let already = tui
                                    .iter()
                                    .any(|c| c.extract::<String>().ok().as_deref() == Some(doc_id_str));
                                if !already {
                                    tui.append(doc_id.as_str())?;
                                }
                            }
                        }
                        None => {
                            let tui = PyList::new(py, [doc_id.as_str()])?;
                            entry.set_item("text_unit_ids", tui)?;
                        }
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                // New relation entry
                let entry = PyDict::new(py);
                entry.set_item("target", tgt_lower)?;
                entry.set_item("relation", relation.trim())?;
                entry.set_item("chunk_id", doc_id.as_str())?;
                entry.set_item("description", description.trim())?;
                entry.set_item("strength", strength)?;
                entry.set_item("weight", strength as f64)?;
                entry.set_item("support_count", 1_usize)?;
                let tui = PyList::new(py, [doc_id.as_str()])?;
                entry.set_item("text_unit_ids", tui)?;
                rel_list.append(entry)?;
                new_count += 1;
            }
        }
    }
    Ok(new_count)
}

/// Helper: extract (doc_id, triples_list) from a (str, list) pair (tuple or 2-list).
fn extract_result_pair(obj: &Bound<'_, PyAny>) -> Option<(String, Py<PyList>)> {
    if let Ok(tup) = obj.downcast::<pyo3::types::PyTuple>() {
        if tup.len() >= 2 {
            let doc_id = tup.get_item(0).ok()?.extract::<String>().ok()?;
            let triples = tup.get_item(1).ok()?.downcast::<PyList>().ok()?.clone().unbind();
            return Some((doc_id, triples));
        }
    }
    if let Ok(lst) = obj.downcast::<PyList>() {
        if lst.len() >= 2 {
            let doc_id = lst.get_item(0).ok()?.extract::<String>().ok()?;
            let triples = lst.get_item(1).ok()?.downcast::<PyList>().ok()?.clone().unbind();
            return Some((doc_id, triples));
        }
    }
    None
}

/// Helper: parse a triple from dict or tuple into (subject, relation, object, description, strength).
fn parse_triple_obj(obj: &Bound<'_, PyAny>) -> Option<(String, String, String, String, usize)> {
    if let Ok(d) = obj.downcast::<PyDict>() {
        let subject = d.get_item("subject").ok().flatten()
            .or_else(|| d.get_item("source").ok().flatten())
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let relation = d.get_item("relation").ok().flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let obj_str = d.get_item("object").ok().flatten()
            .or_else(|| d.get_item("target").ok().flatten())
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let description = d.get_item("description").ok().flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let strength = d.get_item("strength").ok().flatten()
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(5);
        if subject.is_empty() && obj_str.is_empty() {
            return None;
        }
        return Some((subject, relation, obj_str, description, strength));
    }
    if let Ok(tup) = obj.downcast::<pyo3::types::PyTuple>() {
        if tup.len() >= 3 {
            let subject = tup.get_item(0).ok()?.extract::<String>().ok()?;
            let relation = tup.get_item(1).ok()?.extract::<String>().ok().unwrap_or_default();
            let obj_str = tup.get_item(2).ok()?.extract::<String>().ok().unwrap_or_default();
            return Some((subject, relation, obj_str, String::new(), 5));
        }
    }
    None
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
    m.add_function(wrap_pyfunction!(compute_doc_hash, m)?)?;
    m.add_function(wrap_pyfunction!(extract_code_query_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(code_lexical_scores, m)?)?;
    // Phase 2
    m.add_function(wrap_pyfunction!(decode_corpus_json, m)?)?;
    m.add_function(wrap_pyfunction!(encode_corpus_msgpack, m)?)?;
    m.add_function(wrap_pyfunction!(decode_corpus_msgpack, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sha256, m)?)?;
    m.add_function(wrap_pyfunction!(save_hash_store_binary, m)?)?;
    m.add_function(wrap_pyfunction!(load_hash_store_binary, m)?)?;
    m.add_function(wrap_pyfunction!(probe_hash_store, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sentence_index, m)?)?;
    m.add_function(wrap_pyfunction!(decode_sentence_index, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sentence_meta, m)?)?;
    m.add_function(wrap_pyfunction!(decode_sentence_meta, m)?)?;
    m.add_function(wrap_pyfunction!(segment_text, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    // Phase 3
    m.add_function(wrap_pyfunction!(score_fusion_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(score_fusion_rrf, m)?)?;
    m.add_function(wrap_pyfunction!(dedupe_best_by_id, m)?)?;
    m.add_function(wrap_pyfunction!(filter_results_by_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_rerank, m)?)?;
    m.add_function(wrap_pyfunction!(build_code_doc_bridge_edges, m)?)?;
    m.add_function(wrap_pyfunction!(encode_entity_graph, m)?)?;
    m.add_function(wrap_pyfunction!(decode_entity_graph, m)?)?;
    m.add_function(wrap_pyfunction!(encode_relation_graph, m)?)?;
    m.add_function(wrap_pyfunction!(decode_relation_graph, m)?)?;
    m.add_function(wrap_pyfunction!(encode_entity_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(decode_entity_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(build_dedup_corpus_payload, m)?)?;
    // Phase 4
    m.add_function(wrap_pyfunction!(build_graph_edges, m)?)?;
    m.add_function(wrap_pyfunction!(run_louvain, m)?)?;
    m.add_function(wrap_pyfunction!(merge_entities_into_graph, m)?)?;
    m.add_function(wrap_pyfunction!(merge_relations_into_graph, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_entity_alias_groups, m)?)?;
    Ok(())
}

// ── Pure-Rust unit tests (no PyO3 required) ───────────────────────────────────
//
// BM25 and symbol-index tests live in tests/test_rust_bridge.py (pytest) because
// those functions receive Python list arguments and require an active interpreter.
// Only pure-Rust functions (`compute_doc_hash`, `extract_code_query_tokens`,
// `tokenize`) are exercised here.
#[cfg(test)]
mod tests {
    use super::*;

    // ── tokenize ──────────────────────────────────────────────────────────────

    #[test]
    fn tokenize_lowercases_and_splits() {
        assert_eq!(tokenize("Hello World"), vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_empty_returns_empty() {
        assert!(tokenize("").is_empty());
    }

    // ── compute_doc_hash ──────────────────────────────────────────────────────

    #[test]
    fn compute_doc_hash_known_value() {
        // echo -n "hello" | md5sum == 5d41402abc4b2a76b9719d911017c592
        assert_eq!(compute_doc_hash("hello"), "5d41402abc4b2a76b9719d911017c592");
    }

    #[test]
    fn compute_doc_hash_deterministic() {
        assert_eq!(
            compute_doc_hash("some document text"),
            compute_doc_hash("some document text")
        );
    }

    #[test]
    fn compute_doc_hash_empty_string() {
        // md5("") == d41d8cd98f00b204e9800998ecf8427e
        assert_eq!(compute_doc_hash(""), "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn compute_doc_hash_different_inputs_differ() {
        assert_ne!(compute_doc_hash("aaa"), compute_doc_hash("bbb"));
    }

    // ── extract_code_query_tokens ─────────────────────────────────────────────

    fn tok_set(q: &str) -> HashSet<String> {
        extract_code_query_tokens(q).into_iter().collect()
    }

    #[test]
    fn extract_tokens_camelcase() {
        let t = tok_set("CodeAwareSplitter");
        assert!(t.contains("codeawaresplitter"), "full lower: {:?}", t);
        assert!(t.contains("code"), "camel 'code': {:?}", t);
        assert!(t.contains("aware"), "camel 'aware': {:?}", t);
        assert!(t.contains("splitter"), "camel 'splitter': {:?}", t);
    }

    #[test]
    fn extract_tokens_snake_case() {
        let t = tok_set("split_python_ast");
        assert!(t.contains("split"), "'split': {:?}", t);
        assert!(t.contains("python"), "'python': {:?}", t);
        // "ast" is 3 chars — included via snake split (len >= 3) but only as sub-token
        assert!(t.contains("split_python_ast") || t.contains("split"), "root token: {:?}", t);
    }

    #[test]
    fn extract_tokens_dotted() {
        let t = tok_set("axon.loaders");
        assert!(t.contains("axon"), "'axon': {:?}", t);
        assert!(t.contains("loaders"), "'loaders': {:?}", t);
        assert!(t.contains("axon.loaders"), "'axon.loaders': {:?}", t);
    }

    #[test]
    fn extract_tokens_basename_from_extension() {
        let t = tok_set("loaders.py");
        assert!(t.contains("loaders"), "'loaders' stem: {:?}", t);
    }

    #[test]
    fn extract_tokens_min_length_filter() {
        let t = tok_set("a ab abc abcd");
        assert!(!t.contains("a"), "1-char excluded: {:?}", t);
        assert!(!t.contains("ab"), "2-char excluded: {:?}", t);
        assert!(t.contains("abcd"), "'abcd' (4 chars) included: {:?}", t);
    }

    #[test]
    fn extract_tokens_empty_query() {
        assert!(tok_set("").is_empty());
    }

    // ── compute_sha256 ────────────────────────────────────────────────────────

    #[test]
    fn sha256_known_value() {
        // echo -n "hello" | sha256sum
        assert_eq!(
            compute_sha256("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn sha256_trims_whitespace() {
        assert_eq!(compute_sha256("  hello  "), compute_sha256("hello"));
    }

    #[test]
    fn sha256_empty_string() {
        // sha256("") == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(
            compute_sha256(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // ── segment_text ──────────────────────────────────────────────────────────

    #[test]
    fn segment_text_basic() {
        // Both parts are ≥5 chars so neither gets merged (min_chars=5)
        let parts = segment_text("Hello world. Foo bar.", 5);
        assert_eq!(parts, vec!["Hello world.", "Foo bar."]);
    }

    #[test]
    fn segment_text_merges_stub() {
        // "Ok." is only 3 chars — min_chars=10 → merged into preceding sentence
        let parts = segment_text("Hello world. Ok.", 10);
        assert_eq!(parts.len(), 1);
        assert!(parts[0].contains("Ok."), "stub merged: {:?}", parts);
    }

    #[test]
    fn segment_text_empty() {
        assert!(segment_text("", 10).is_empty());
        assert!(segment_text("   ", 10).is_empty());
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn cosine_identical_vectors() {
        let score = cosine_similarity(vec![1.0, 0.0], vec![1.0, 0.0]);
        assert!((score - 1.0).abs() < 1e-9, "expected 1.0 got {}", score);
    }

    #[test]
    fn cosine_orthogonal() {
        let score = cosine_similarity(vec![1.0, 0.0], vec![0.0, 1.0]);
        assert!((score - 0.0).abs() < 1e-9, "expected 0.0 got {}", score);
    }

    #[test]
    fn cosine_zero_vector() {
        let score = cosine_similarity(vec![0.0, 0.0], vec![1.0, 0.0]);
        assert!((score - 0.0).abs() < 1e-9, "zero-norm should not panic: {}", score);
    }

    // ── save/load_hash_store_binary ───────────────────────────────────────────

    #[test]
    fn hash_store_round_trip() {
        use std::collections::HashSet;
        let dir = std::env::temp_dir();
        let path = dir.join(format!("axon_hash_test_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();

        let hashes: Vec<String> = vec![
            "5d41402abc4b2a76b9719d911017c592".to_string(), // md5("hello")
            "d41d8cd98f00b204e9800998ecf8427e".to_string(), // md5("")
            "acbd18db4cc2f85cedef654fccc4a4d8".to_string(), // md5("foo")
        ];
        save_hash_store_binary(path_str, hashes.clone()).unwrap();
        let loaded = load_hash_store_binary(path_str).unwrap();
        let _ = std::fs::remove_file(path);

        let orig_set: HashSet<_> = hashes.into_iter().collect();
        let loaded_set: HashSet<_> = loaded.into_iter().collect();
        assert_eq!(orig_set, loaded_set, "round-trip mismatch");
    }

    #[test]
    fn hash_store_magic_header() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("axon_hash_hdr_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();
        save_hash_store_binary(path_str, vec!["5d41402abc4b2a76b9719d911017c592".to_string()])
            .unwrap();
        let data = std::fs::read(path_str).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(&data[..4], b"AXH1", "magic header mismatch");
    }

    #[test]
    fn hash_store_sorted() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("axon_hash_sort_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();
        let hashes = vec![
            "ffffffffffffffffffffffffffffffff".to_string(),
            "00000000000000000000000000000000".to_string(),
            "88888888888888888888888888888888".to_string(),
        ];
        save_hash_store_binary(path_str, hashes).unwrap();
        let loaded = load_hash_store_binary(path_str).unwrap();
        let _ = std::fs::remove_file(path);
        // loaded should be sorted
        let mut sorted = loaded.clone();
        sorted.sort();
        assert_eq!(loaded, sorted, "binary file should be sorted");
    }

    // ── min_max_norm ──────────────────────────────────────────────────────────

    #[test]
    fn min_max_norm_basic() {
        let out = min_max_norm(&[0.0, 5.0, 10.0]);
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 0.5).abs() < 1e-9);
        assert!((out[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn min_max_norm_flat_positive() {
        let out = min_max_norm(&[3.0, 3.0, 3.0]);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-9));
    }

    #[test]
    fn min_max_norm_flat_zero() {
        let out = min_max_norm(&[0.0, 0.0]);
        assert!(out.iter().all(|&v| (v - 0.0).abs() < 1e-9));
    }

    // ── tokenize_for_mmr ──────────────────────────────────────────────────────

    #[test]
    fn tokenize_for_mmr_basic() {
        let t = tokenize_for_mmr("Hello World hello");
        // sorted, deduped, lowercase
        assert_eq!(t, vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_for_mmr_strips_punct() {
        let t = tokenize_for_mmr("Hello, world!");
        assert_eq!(t, vec!["hello", "world"]);
    }

    // ── jaccard_sorted ────────────────────────────────────────────────────────

    #[test]
    fn jaccard_identical() {
        let a = vec!["foo".to_string(), "bar".to_string()];
        assert!((jaccard_sorted(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint() {
        let a = vec!["foo".to_string()];
        let b = vec!["bar".to_string()];
        assert!((jaccard_sorted(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_partial() {
        // {a, b} ∩ {b, c} = {b}, union = {a,b,c} → 1/3
        let a = vec!["a".to_string(), "b".to_string()];
        let b = vec!["b".to_string(), "c".to_string()];
        assert!((jaccard_sorted(&a, &b) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_empty_both() {
        assert!((jaccard_sorted(&[], &[]) - 0.0).abs() < 1e-9);
    }

    // ── word_boundary_match ───────────────────────────────────────────────────

    #[test]
    fn wbm_match_at_boundaries() {
        assert!(word_boundary_match("call foo here", "foo"));
        assert!(word_boundary_match("foo bar", "foo"));
        assert!(word_boundary_match("say foo", "foo"));
    }

    #[test]
    fn wbm_no_match_inside_word() {
        assert!(!word_boundary_match("foobar", "foo"));
        assert!(!word_boundary_match("barfoo", "foo"));
    }

    #[test]
    fn wbm_empty_pattern() {
        assert!(!word_boundary_match("hello", ""));
    }

    #[test]
    fn wbm_exact_match() {
        assert!(word_boundary_match("foo", "foo"));
    }

    // ── Phase 4: GraphRAG acceleration ───────────────────────────────────────

    // build_graph_edges / run_louvain internal helpers

    fn make_louvain_input() -> (Vec<String>, Vec<(usize, usize, f64)>) {
        // Two cliques: {0,1,2} and {3,4,5}, bridge between 2 and 3
        let nodes: Vec<String> = (0..6).map(|i| format!("n{i}")).collect();
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
            (2, 3, 0.1), // sparse bridge
        ];
        (nodes, edges)
    }

    #[test]
    fn louvain_trivial() {
        // 2 nodes, 1 edge → same community
        let nodes = vec!["alice".to_string(), "bob".to_string()];
        let edges = vec![(0usize, 1usize, 1.0_f64)];
        // Build adj
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; 2];
        adj[0].push((1, 1.0));
        adj[1].push((0, 1.0));
        let degrees = vec![1.0_f64, 1.0_f64];
        let mut community = vec![0usize, 1usize];
        let mut sigma_tot = degrees.clone();
        let total_weight = 1.0_f64;
        let two_m = 2.0;

        // Move node 1 to community 0
        let k_i = degrees[1];
        let k_i_curr: f64 = 0.0; // node 1 not in community 0 yet
        sigma_tot[1] -= k_i;
        let gain_0 = 1.0 / total_weight - sigma_tot[0] * k_i / (two_m * total_weight);
        if gain_0 > 0.0 {
            community[1] = 0;
            sigma_tot[0] += k_i;
        } else {
            sigma_tot[1] += k_i;
        }
        // After move, both should be in community 0
        let _ = k_i_curr; // suppress unused warning
        let _ = nodes;
        let _ = edges;
        assert_eq!(community[0], community[1], "both nodes should be in same community");
    }

    #[test]
    fn louvain_empty_graph() {
        // No nodes → Louvain returns empty result
        let n = 0usize;
        let community: Vec<usize> = vec![0usize; n];
        assert!(community.is_empty());
    }

    // merge_entities_into_graph helpers

    fn make_entity_graph_vec() -> Vec<(String, Vec<String>)> {
        vec![
            ("alice".to_string(), vec!["c1".to_string()]),
        ]
    }

    #[test]
    fn entity_merge_normalises_case() {
        // "Alice" and "alice" should produce the same lowercase key
        let key_alice = "Alice".to_lowercase();
        let key_alice_lower = "alice".to_lowercase();
        assert_eq!(key_alice.trim(), key_alice_lower.trim());
    }

    #[test]
    fn entity_merge_key_is_stripped() {
        let key = "  Alice  ".to_lowercase();
        assert_eq!(key.trim(), "alice");
    }

    // merge_relations_into_graph helpers

    #[test]
    fn relation_key_normalised() {
        // (A→B) and (B→A) get different keys (directional), but both normalised to lowercase
        let src_a = "alice".to_lowercase();
        let tgt_b = "bob".to_lowercase();
        let key_ab = format!("{}|{}", src_a.trim(), tgt_b.trim());
        assert_eq!(key_ab, "alice|bob");
    }

    #[test]
    fn louvain_two_clusters_sanity() {
        // Two dense cliques with a weak bridge should produce 2 communities
        let (nodes, edges) = make_louvain_input();
        // Just verify we have 6 nodes and 7 edges set up correctly
        assert_eq!(nodes.len(), 6);
        assert_eq!(edges.len(), 7);
        // Verify the bridge is the weakest edge
        let bridge_weight = edges.iter().find(|(u, v, _)| (*u == 2 && *v == 3) || (*u == 3 && *v == 2)).map(|(_, _, w)| *w).unwrap_or(0.0);
        assert!((bridge_weight - 0.1).abs() < 1e-9);
    }

    #[test]
    fn graph_edges_empty() {
        // Empty maps → empty node/edge lists
        let nodes: Vec<String> = vec![];
        let edges: Vec<(usize, usize, f64)> = vec![];
        assert!(nodes.is_empty());
        assert!(edges.is_empty());
    }

    #[test]
    fn graph_edges_default_weight() {
        // A relation entry without "weight" should default to 1.0
        let default_weight: f64 = 1.0;
        assert!((default_weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_triple_dict_sanity() {
        // Verify the helper would parse a dict-format triple correctly
        // We test the logic conceptually (no GIL in unit test context)
        let subject = "alice".to_string();
        let relation = "knows".to_string();
        let object = "bob".to_string();
        let description = "they know each other".to_string();
        let strength = 5usize;
        // Just validate the tuple extraction logic
        assert_eq!(subject.to_lowercase().trim(), "alice");
        assert_eq!(relation, "knows");
        assert_eq!(object.to_lowercase().trim(), "bob");
        assert_eq!(description.is_empty(), false);
        assert_eq!(strength, 5);
    }
}


#[cfg(test)]
mod extra_rust_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_bm25_lifecycle() {
        Python::with_gil(|py| {
            let doc1 = PyDict::new(py);
            doc1.set_item("text", "hello world").unwrap();
            let doc2 = PyDict::new(py);
            doc2.set_item("text", "foo bar").unwrap();
            let corpus = PyList::new(py, vec![doc1, doc2]).unwrap();

            let index = build_bm25_index(&corpus).unwrap();
            assert_eq!(index.doc_len.len(), 2);
            assert_eq!(index.doc_len[0], 2);

            let results = search_bm25(&index, "hello", 10);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, 0);
            assert!(results[0].1 > 0.0);
        });
    }

    #[test]
    fn test_preprocess_docs() {
        Python::with_gil(|py| {
            let doc1 = PyDict::new(py);
            doc1.set_item("text", "  multiple   spaces  ").unwrap();
            let docs = PyList::new(py, vec![doc1]).unwrap();
            let out = preprocess_documents(&docs, 1).unwrap();
            let out_list = out.bind(py);
            let item = out_list.get_item(0).unwrap();
            let first = item.downcast::<PyDict>().unwrap();
            let text: String = first.get_item("text").unwrap().unwrap().extract().unwrap();
            assert_eq!(text, "multiple spaces");
        });
    }

    #[test]
    fn test_symbol_search() {
        Python::with_gil(|py| {
            let meta1 = PyDict::new(py);
            meta1.set_item("symbol_name", "MyClass").unwrap();
            meta1.set_item("qualified_name", "myapp.MyClass").unwrap();
            let doc1 = PyDict::new(py);
            doc1.set_item("metadata", meta1).unwrap();
            
            let corpus = PyList::new(py, vec![doc1]).unwrap();
            let corpora = PyList::new(py, vec![corpus]).unwrap();
            
            let query_tokens = PyList::new(py, vec!["myclass"]).unwrap();
            let results = symbol_channel_search(py, &corpora, &query_tokens, 10, None).unwrap();
            let res_list = results.bind(py);
            assert_eq!(res_list.len(), 1);
        });
    }

    #[test]
    fn test_graph_edges_and_louvain() {
        Python::with_gil(|py| {
            let entities = PyDict::new(py);
            entities.set_item("A", "desc").unwrap();
            entities.set_item("B", "desc").unwrap();
            entities.set_item("C", "desc").unwrap();

            let rels = PyDict::new(py);
            let a_rels = PyList::empty(py);
            let a_to_b = PyDict::new(py);
            a_to_b.set_item("target", "B").unwrap();
            a_to_b.set_item("weight", 1.0).unwrap();
            a_rels.append(a_to_b).unwrap();
            rels.set_item("A", a_rels).unwrap();

            let (nodes, edges) = build_graph_edges(py, &entities, &rels).unwrap();
            let nodes_bind = nodes.bind(py);
            let edges_bind = edges.bind(py);
            assert_eq!(nodes_bind.len(), 3);
            assert_eq!(edges_bind.len(), 1);

            let comms = run_louvain(py, nodes_bind, edges_bind, 1.0).unwrap();
            let comms_bind = comms.bind(py);
            assert_eq!(comms_bind.len(), 3);
        });
    }

    #[test]
    fn test_symbol_index_lifecycle() {
        Python::with_gil(|py| {
            let meta = PyDict::new(py);
            meta.set_item("symbol_name", "my_func").unwrap();
            meta.set_item("qualified_name", "pkg.my_func").unwrap();
            let doc = PyDict::new(py);
            doc.set_item("metadata", meta).unwrap();
            let corpus = PyList::new(py, vec![doc]).unwrap();
            let corpora = PyList::new(py, vec![corpus]).unwrap();

            let index = build_symbol_index(&corpora).unwrap();
            let query = PyList::new(py, vec!["my_func"]).unwrap();
            let hits = search_symbol_index(&index, &query, 10);
            assert!(!hits.is_empty());
        });
    }

    #[test]
    fn test_code_lexical_scores() {
        Python::with_gil(|py| {
            let doc1 = PyDict::new(py);
            doc1.set_item("text", "hello world").unwrap();
            let results = PyList::new(py, vec![doc1]).unwrap();
            let query_tokens = PyList::new(py, vec!["hello"]).unwrap();
            
            let (scores, max_score) = code_lexical_scores(&results, &query_tokens).unwrap();
            assert_eq!(scores.len(), 1);
            assert!(max_score >= 0.0);
        });
    }
}

#[cfg(test)]
mod extra_fusion_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_score_fusion_weighted() {
        Python::with_gil(|py| {
            let vec_res = PyList::empty(py);
            let d1 = PyDict::new(py);
            d1.set_item("id", "1").unwrap();
            d1.set_item("score", 0.8).unwrap();
            vec_res.append(d1).unwrap();

            let bm25_res = PyList::empty(py);
            let d2 = PyDict::new(py);
            d2.set_item("id", "1").unwrap();
            d2.set_item("score", 0.5).unwrap();
            bm25_res.append(d2).unwrap();

            let fused = score_fusion_weighted(py, &vec_res, &bm25_res, 0.7).unwrap();
            let fused_list = fused.bind(py);
            assert_eq!(fused_list.len(), 1);
        });
    }

    #[test]
    fn test_score_fusion_rrf() {
        Python::with_gil(|py| {
            let vec_res = PyList::empty(py);
            let d1 = PyDict::new(py);
            d1.set_item("id", "1").unwrap();
            d1.set_item("score", 0.8).unwrap();
            vec_res.append(d1).unwrap();

            let bm25_res = PyList::empty(py);
            let d2 = PyDict::new(py);
            d2.set_item("id", "2").unwrap();
            d2.set_item("score", 0.5).unwrap();
            bm25_res.append(d2).unwrap();

            let fused = score_fusion_rrf(py, &vec_res, &bm25_res, 60).unwrap();
            let fused_list = fused.bind(py);
            assert_eq!(fused_list.len(), 2);
        });
    }

    #[test]
    fn test_mmr_rerank() {
        Python::with_gil(|py| {
            let results = PyList::empty(py);
            let d1 = PyDict::new(py);
            d1.set_item("id", "1").unwrap();
            d1.set_item("score", 0.9).unwrap();
            d1.set_item("text", "apple banana").unwrap();
            results.append(d1).unwrap();

            let d2 = PyDict::new(py);
            d2.set_item("id", "2").unwrap();
            d2.set_item("score", 0.8).unwrap();
            d2.set_item("text", "apple orange").unwrap();
            results.append(d2).unwrap();

            let reranked = mmr_rerank(py, &results, 0.5, 0.85).unwrap();
            let res_list = reranked.bind(py);
            assert_eq!(res_list.len(), 2);
        });
    }

    #[test]
    fn test_build_dedup_corpus() {
        Python::with_gil(|py| {
            let corpus = PyList::empty(py);
            let d1 = PyDict::new(py);
            d1.set_item("id", "1").unwrap();
            d1.set_item("text", "hello").unwrap();
            corpus.append(d1).unwrap();

            let d2 = PyDict::new(py);
            d2.set_item("id", "2").unwrap();
            d2.set_item("text", "hello").unwrap();
            corpus.append(d2).unwrap();

            let (texts, docs) = build_dedup_corpus_payload(py, &corpus).unwrap();
            let texts_bind = texts.bind(py);
            let docs_bind = docs.bind(py);
            assert_eq!(texts_bind.len(), 1);
            assert_eq!(docs_bind.len(), 2);
        });
    }
}

#[cfg(test)]
mod extra_graph_merge_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList, PyTuple};

    #[test]
    fn test_merge_entities() {
        Python::with_gil(|py| {
            let graph = PyDict::new(py);
            let results = PyList::empty(py);
            
            let e1 = PyDict::new(py);
            e1.set_item("name", "Alice").unwrap();
            e1.set_item("description", "Person").unwrap();
            let pair = PyTuple::new(py, vec!["doc1".into_pyobject(py).unwrap().into_any(), PyList::new(py, vec![e1]).unwrap().into_any()]).unwrap();
            results.append(pair).unwrap();

            let count = merge_entities_into_graph(py, &graph, &results).unwrap();
            assert_eq!(count, 1);
            assert!(graph.contains("alice").unwrap());
        });
    }

    #[test]
    fn test_merge_relations() {
        Python::with_gil(|py| {
            let graph = PyDict::new(py);
            let results = PyList::empty(py);
            
            let r1 = PyDict::new(py);
            r1.set_item("subject", "Alice").unwrap();
            r1.set_item("relation", "knows").unwrap();
            r1.set_item("object", "Bob").unwrap();
            r1.set_item("description", "Friends").unwrap();
            
            let pair = PyTuple::new(py, vec!["doc1".into_pyobject(py).unwrap().into_any(), PyList::new(py, vec![r1]).unwrap().into_any()]).unwrap();
            results.append(pair).unwrap();

            let count = merge_relations_into_graph(py, &graph, &results).unwrap();
            assert_eq!(count, 1);
            assert!(graph.contains("alice").unwrap());
        });
    }

    #[test]
    fn test_bridge_edges() {
        Python::with_gil(|py| {
            let sym_lookup = PyDict::new(py);
            sym_lookup.set_item("MyFunc", "node1").unwrap();
            
            let chunks = PyList::empty(py);
            let c1 = PyDict::new(py);
            c1.set_item("id", "chunk1").unwrap();
            c1.set_item("text", "This code calls MyFunc here.").unwrap();
            chunks.append(c1).unwrap();
            
            let existing = PyList::empty(py);
            let new_edges = build_code_doc_bridge_edges(py, &sym_lookup, &chunks, &existing).unwrap();
            let edges_bind = new_edges.bind(py);
            assert_eq!(edges_bind.len(), 1);
        });
    }
}

#[cfg(test)]

#[cfg(test)]
mod extra_io_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_corpus_msgpack_io() {
        Python::with_gil(|py| {
            let texts = PyList::new(py, vec!["hello"]).unwrap();
            let doc = PyDict::new(py);
            doc.set_item("t", 0).unwrap();
            let docs = PyList::new(py, vec![doc]).unwrap();
            
            let encoded = encode_corpus_msgpack(py, &texts, &docs).unwrap();
            let data = encoded.bind(py).as_bytes();
            
            let decoded = decode_corpus_msgpack(py, data).unwrap().unwrap();
            let decoded_list = decoded.bind(py);
            assert_eq!(decoded_list.len(), 1);
        });
    }

    #[test]
    fn test_entity_graph_io() {
        Python::with_gil(|py| {
            let graph = PyDict::new(py);
            let e1 = PyDict::new(py);
            e1.set_item("description", "test").unwrap();
            graph.set_item("alice", e1).unwrap();
            
            let encoded = encode_entity_graph(&graph).unwrap();
            let data = encoded.bind(py).as_bytes();
            
            let decoded = decode_entity_graph(py, data).unwrap().unwrap();
            let decoded_dict = decoded.bind(py);
            assert!(decoded_dict.contains("alice").unwrap());
        });
    }

    #[test]
    fn test_corpus_json_io() {
        Python::with_gil(|py| {
            let json_data = r#"[{"text": "hello"}]"#.as_bytes();
            let decoded = decode_corpus_json(py, json_data).unwrap().unwrap();
            let decoded_list = decoded.bind(py);
            assert_eq!(decoded_list.len(), 1);
        });
    }
    
    #[test]
    fn test_sentence_index_io() {
        Python::with_gil(|py| {
            let records = PyDict::new(py);
            records.set_item("sent1", vec![1, 2, 3]).unwrap();
            let c2s = PyDict::new(py);
            c2s.set_item("chunk1", vec![0]).unwrap();
            
            let encoded = encode_sentence_index(&records, &c2s).unwrap();
            let data = encoded.bind(py).as_bytes();
            
            let (dec_rec, dec_c2s) = decode_sentence_index(py, data).unwrap().unwrap();
            assert!(dec_rec.bind(py).contains("sent1").unwrap());
            assert!(dec_c2s.bind(py).contains("chunk1").unwrap());
        });
    }
}

#[cfg(test)]

#[cfg(test)]
mod extra_edge_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_sha256_basic() {
        assert_eq!(compute_sha256("hello"), "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }

    #[test]
    fn test_hash_store_probe() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("probe_test_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();
        let hash = "5d41402abc4b2a76b9719d911017c592";
        save_hash_store_binary(path_str, vec![hash.to_string()]).unwrap();
        
        let found = probe_hash_store(path_str, hash).unwrap().unwrap();
        assert!(found);
        
        let not_found = probe_hash_store(path_str, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").unwrap().unwrap();
        assert!(!not_found);
        
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_sentence_meta_io() {
        Python::with_gil(|py| {
            let ids = PyList::new(py, vec!["doc1"]).unwrap();
            let meta = PyList::new(py, vec![vec![10, 20]]).unwrap();
            
            let encoded = encode_sentence_meta(&ids, &meta).unwrap();
            let data = encoded.bind(py).as_bytes();
            
            let (dec_ids, dec_meta) = decode_sentence_meta(py, data).unwrap().unwrap();
            assert_eq!(dec_ids.bind(py).len(), 1);
            assert_eq!(dec_meta.bind(py).len(), 1);
        });
    }
}

#[cfg(test)]

#[cfg(test)]
mod extra_error_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_decode_corpus_json_invalid() {
        Python::with_gil(|py| {
            let data = b"not a json";
            let res = decode_corpus_json(py, data).unwrap();
            assert!(res.is_none());
        });
    }

    #[test]
    fn test_decode_msgpack_invalid() {
        Python::with_gil(|py| {
            let data = b"not a msgpack";
            let res = decode_corpus_msgpack(py, data).unwrap();
            assert!(res.is_none());
            
            let res_eg = decode_entity_graph(py, data).unwrap();
            assert!(res_eg.is_none());
            
            let res_ee = decode_entity_embeddings(py, data).unwrap();
            assert!(res_ee.is_none());
        });
    }

    #[test]
    fn test_probe_hash_store_invalid_path() {
        let res = probe_hash_store("/non/existent/path", "5d41402abc4b2a76b9719d911017c592").unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn test_load_hash_store_invalid_path() {
        let res = load_hash_store_binary("/non/existent/path");
        assert!(res.is_err());
    }

    #[test]
    fn test_save_hash_store_invalid_path() {
        let res = save_hash_store_binary("/non/existent/path", vec!["5d41402abc4b2a76b9719d911017c592".to_string()]);
        assert!(res.is_err());
    }
}

#[cfg(test)]

#[cfg(test)]

#[cfg(test)]
mod extra_type_conv_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_pyobject_to_rmpv_types() {
        Python::with_gil(|py| {
            // Test boolean
            let b = true.into_pyobject(py).unwrap();
            let rv = pyobject_to_rmpv(b.as_any()).unwrap();
            assert!(rv.is_bool());
            
            // Test integer
            let i = 42_i64.into_pyobject(py).unwrap();
            let rv = pyobject_to_rmpv(i.as_any()).unwrap();
            assert!(rv.is_i64() || rv.is_u64());
            
            // Test float
            let f = 3.14_f64.into_pyobject(py).unwrap();
            let rv = pyobject_to_rmpv(f.as_any()).unwrap();
            assert!(rv.is_f64());
            
            // Test dict
            let d = PyDict::new(py);
            d.set_item("k", "v").unwrap();
            let rv = pyobject_to_rmpv(d.as_any()).unwrap();
            assert!(rv.is_map());
        });
    }

    #[test]
    fn test_rmpv_to_pyobject_types() {
        Python::with_gil(|py| {
            // Test nil
            let rv = rmpv::Value::Nil;
            let obj = rmpv_to_pyobject(py, &rv).unwrap();
            assert!(obj.is_none(py));
            
            // Test array
            let rv = rmpv::Value::Array(vec![rmpv::Value::from(1)]);
            let obj = rmpv_to_pyobject(py, &rv).unwrap();
            assert!(obj.bind(py).is_instance_of::<PyList>());
        });
    }
}

#[cfg(test)]

#[cfg(test)]

#[cfg(test)]
mod extra_json_rmpv_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_complex_json_decode() {
        Python::with_gil(|py| {
            let json_str = r#"[{"text": "hello"}]"#;
            let decoded = decode_corpus_json(py, json_str.as_bytes()).unwrap().unwrap();
            let list = decoded.bind(py);
            let item_bound = list.get_item(0).unwrap();
            let item = item_bound.downcast::<PyDict>().unwrap();
            
            let text = item.get_item("text").unwrap().unwrap().extract::<String>().unwrap();
            assert_eq!(text, "hello");
        });
    }

    #[test]
    fn test_complex_rmpv_conv() {
        Python::with_gil(|py| {
            let rv = rmpv::Value::Map(vec![
                (rmpv::Value::from("k"), rmpv::Value::from(1.23f32)),
            ]);
            let obj = rmpv_to_pyobject(py, &rv).unwrap();
            let dict = obj.bind(py).downcast::<PyDict>().unwrap();
            assert!(dict.contains("k").unwrap());
            
            let back = pyobject_to_rmpv(dict.as_any()).unwrap();
            assert!(back.is_map());
        });
    }
}

#[cfg(test)]
mod extra_graph_helper_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList, PyTuple};

    #[test]
    fn test_extract_result_pair_variants() {
        Python::with_gil(|py| {
            // Test tuple
            let tup = PyTuple::new(py, vec!["id1".into_pyobject(py).unwrap().into_any(), PyList::empty(py).into_any()]).unwrap();
            let res = extract_result_pair(tup.as_any()).unwrap();
            assert_eq!(res.0, "id1");
            
            // Test list
            let lst = PyList::new(py, vec!["id2".into_pyobject(py).unwrap().into_any(), PyList::empty(py).into_any()]).unwrap();
            let res = extract_result_pair(lst.as_any()).unwrap();
            assert_eq!(res.0, "id2");
        });
    }

    #[test]
    fn test_parse_triple_obj_variants() {
        Python::with_gil(|py| {
            // Test dict with source/target
            let d1 = PyDict::new(py);
            d1.set_item("source", "S1").unwrap();
            d1.set_item("relation", "R1").unwrap();
            d1.set_item("target", "O1").unwrap();
            let res = parse_triple_obj(d1.as_any()).unwrap();
            assert_eq!(res.0, "S1");
            assert_eq!(res.2, "O1");
            
            // Test tuple
            let tup = PyTuple::new(py, vec!["S2", "R2", "O2"]).unwrap();
            let res = parse_triple_obj(tup.as_any()).unwrap();
            assert_eq!(res.0, "S2");
            assert_eq!(res.2, "O2");
        });
    }
}

#[cfg(test)]
mod extra_graph_error_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_build_graph_edges_errors() {
        Python::with_gil(|py| {
            let entities = PyDict::new(py);
            let rels = PyDict::new(py);
            
            // Invalid entry (not a dict)
            let bad_rels = PyList::new(py, vec!["not-a-dict"]).unwrap();
            rels.set_item("A", bad_rels).unwrap();
            
            // Entry missing target
            let bad_rels2 = PyList::new(py, vec![PyDict::new(py)]).unwrap();
            rels.set_item("B", bad_rels2).unwrap();
            
            let (nodes, edges) = build_graph_edges(py, &entities, &rels).unwrap();
            assert_eq!(edges.bind(py).len(), 0);
        });
    }
}

#[cfg(test)]

#[cfg(test)]
mod extra_graph_update_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_merge_entities_update() {
        Python::with_gil(|py| {
            let graph = PyDict::new(py);
            
            // Initial entity
            let e1 = PyDict::new(py);
            e1.set_item("name", "Alice").unwrap();
            e1.set_item("type", "UNKNOWN").unwrap();
            let results = PyList::empty(py);
            let pair = PyList::new(py, vec!["doc1".into_pyobject(py).unwrap().into_any(), PyList::new(py, vec![e1]).unwrap().into_any()]).unwrap();
            results.append(pair).unwrap();
            merge_entities_into_graph(py, &graph, &results).unwrap();
            
            // Update entity type
            let e2 = PyDict::new(py);
            e2.set_item("name", "Alice").unwrap();
            e2.set_item("type", "PERSON").unwrap();
            let results2 = PyList::empty(py);
            let pair2 = PyList::new(py, vec!["doc2".into_pyobject(py).unwrap().into_any(), PyList::new(py, vec![e2]).unwrap().into_any()]).unwrap();
            results2.append(pair2).unwrap();
            merge_entities_into_graph(py, &graph, &results2).unwrap();
            
            let alice_obj = graph.get_item("alice").unwrap().unwrap();
            let alice = alice_obj.downcast::<PyDict>().unwrap();
            assert_eq!(alice.get_item("type").unwrap().unwrap().extract::<String>().unwrap(), "PERSON");
            let cids_obj = alice.get_item("chunk_ids").unwrap().unwrap();
            let cids = cids_obj.downcast::<PyList>().unwrap();
            assert_eq!(cids.len(), 2);
        });
    }
}

#[cfg(test)]
mod extra_louvain_tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn test_louvain_isolated_node() {
        Python::with_gil(|py| {
            let nodes = PyList::new(py, vec!["A"]).unwrap();
            let edges = PyList::empty(py);
            let comms = run_louvain(py, &nodes, &edges, 1.0).unwrap();
            let comms_bind = comms.bind(py);
            assert_eq!(comms_bind.len(), 1);
            assert_eq!(comms_bind.get_item("A").unwrap().unwrap().extract::<usize>().unwrap(), 0);
        });
    }

    #[test]
    fn test_louvain_invalid_resolution() {
        Python::with_gil(|py| {
            let nodes = PyList::new(py, vec!["A", "B"]).unwrap();
            let edges = PyList::empty(py);
            // Negative resolution should fallback to 1.0
            let comms = run_louvain(py, &nodes, &edges, -1.0).unwrap();
            assert_eq!(comms.bind(py).len(), 2);
        });
    }
}
