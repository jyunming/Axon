# Web Search Setup — Brave Search Integration

Axon can fall back to live web search when your local knowledge base doesn't have a confident answer. This is powered by the [Brave Search API](https://brave.com/search/api/).

---

## What it does

When web search is enabled, Axon watches the confidence of every retrieval. If the local knowledge base returns weak or no results, instead of guessing or refusing, Axon fetches a handful of web snippets from Brave Search and uses those to answer.

You will see a `🌐` prefix in the REPL and `"answer_source": "web_snippet_fallback"` in the API response when this happens.

There are two ways web search can trigger:

| Mode | How it triggers | What to enable |
|------|----------------|----------------|
| **Always-on fallback** | Fires any time local retrieval returns no results at all | `web_search.enabled: true` |
| **CRAG-Lite smart fallback** | Also fires when local results exist but look low-confidence (score below a threshold) | `web_search.enabled: true` + `rag.crag_lite: true` |

> **Is Brave required?** Yes. Web search only works with a Brave Search API key. Without it, Axon falls back to `discussion_fallback` (the LLM answers from general knowledge with a disclaimer) or refuses, depending on your config.

---

## Step 1 — Get a Brave Search API key

1. Go to **[https://brave.com/search/api/](https://brave.com/search/api/)**
2. Click **Get Started for Free**
3. Create an account (email + password)
4. On the dashboard, click **Add Subscription** → choose **Data for AI** (free tier available)
5. Your API key appears under **API Keys** — it looks like `BSA...` (a long string)

**Free tier limits:** 2,000 queries per month. Paid plans start at $3/1,000 queries.

---

## Step 2 — Add the key to Axon

Pick one of the three methods:

### Option A — config.yaml (recommended, permanent)

Open `config.yaml` in your Axon folder and add:

```yaml
web_search:
  enabled: true
  brave_api_key: BSAabc123...your-key-here...
```

### Option B — environment variable

Set `BRAVE_API_KEY` before starting Axon:

**Linux / macOS:**
```bash
export BRAVE_API_KEY=BSAabc123...your-key-here...
axon
```

**Windows (PowerShell):**
```powershell
$env:BRAVE_API_KEY = "BSAabc123...your-key-here..."
axon
```

To make it permanent, add the `export` line to your `~/.bashrc` or `~/.zshrc`, or set it in your PowerShell profile.

### Option C — .env file

Create a `.env` file in the Axon root folder (same folder as `config.yaml`):

```env
BRAVE_API_KEY=BSAabc123...your-key-here...
```

Axon loads this file automatically on startup.

### Option D — REPL interactive prompt

Inside the REPL, type:
```
/keys set brave
```
Axon will prompt you to enter the key. It is saved to `~/.axon/` so you don't need to re-enter it next time.

---

## Step 3 — Enable web search in config.yaml

The `brave_api_key` alone is not enough — you also need to tell Axon to use web search:

```yaml
web_search:
  enabled: true                   # turn on web fallback
  brave_api_key: your-key-here    # or use BRAVE_API_KEY env var instead
```

**Full recommended config** (with CRAG-Lite for smarter triggering):

```yaml
web_search:
  enabled: true
  brave_api_key: your-key-here

rag:
  crag_lite: true                         # enable smart confidence scoring
  crag_lite_confidence_threshold: 0.4     # trigger web search when confidence < 0.4
                                          # lower = less aggressive (fires only on very weak results)
                                          # higher = more aggressive (fires more often)
```

---

## Step 4 — Toggle from the REPL at runtime

You don't need to restart Axon to turn web search on or off:

```
axon> /search
```

This toggles `truth_grounding` on or off for the current session. The REPL shows the current state in the status line:

```
search:ON  (Brave Search)
```

or

```
search:off
```

> **If the key is missing:** Axon will warn you when you try to enable `/search` without a configured API key, and stay in the off state.

---

## Step 5 — Verify it is working

**In the REPL:** Ask a question about something that is NOT in your knowledge base. If web search fires, you will see a `🌐` icon and a note that the answer came from web results rather than your documents.

**Via the API:** Send a query and check the `provenance` field in the response:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the latest news about AI?", "include_diagnostics": true}'
```

Look for:

```json
{
  "provenance": {
    "answer_source": "web_snippet_fallback",
    "retrieved_count": 0,
    "web_count": 3
  }
}
```

`answer_source` values:

| Value | Meaning |
|-------|---------|
| `local_kb` | Answer came from your knowledge base |
| `web_snippet_fallback` | Local retrieval was weak or empty — Brave web results were used |
| `no_context_fallback` | No retrieval and no web; LLM answered from general knowledge (`discussion_fallback: true`) |
| `no_results` | Nothing found and strict mode returned a refusal |

---

## Limitations

- **Web snippets, not full pages** — Brave returns short excerpts, not complete articles. Axon answers based on those snippets, so answers may be less detailed than a full document retrieval.
- **Not a fact-checker** — Brave's ranking is based on relevance scores, not factual verification. Treat web-sourced answers with the same critical eye you would any web search result.
- **Rate limits** — The free tier allows 2,000 queries/month total across all your Axon queries that trigger a fallback. CRAG-Lite helps conserve quota by only calling Brave when the local knowledge base genuinely falls short.
- **Offline mode** — Web search is automatically disabled when `offline.enabled: true` is set. See [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md).
- **API failure / timeout** — If the Brave API is unreachable or returns an error, Axon falls back silently: if `discussion_fallback: true` is set, the LLM answers from general knowledge with a disclaimer; otherwise the response indicates no results were found. You will not see a `🌐` icon in this case. Check `axon-api` logs for `BraveSearch error` messages if you suspect fallback is not working.

---

## Troubleshooting

**Web search toggle is on but no 🌐 results appear**

The API key is missing or invalid. Check:
1. `BRAVE_API_KEY` is set correctly — no extra spaces, correct value
2. `web_search.enabled: true` is in `config.yaml`
3. Run `/keys` in the REPL — it shows whether the Brave key is set

**Brave results appear even for questions your KB can answer**

Lower the `crag_lite_confidence_threshold` (e.g. from `0.4` to `0.2`) so the web fallback is less aggressive. Or disable CRAG-Lite entirely and rely on no-results-only fallback:

```yaml
rag:
  crag_lite: false   # only fall back to web if local returns zero results
```

**`429 Too Many Requests` from Brave**

You have hit your monthly query limit. Upgrade your Brave plan, or reduce fallback frequency by raising the confidence threshold.

---

*See [ADVANCED_RAG.md](ADVANCED_RAG.md) for a deeper explanation of how CRAG-Lite confidence scoring works.*
*See [ADMIN_REFERENCE.md § 6.5](ADMIN_REFERENCE.md) for the full config reference.*
