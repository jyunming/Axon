"""
Tool definitions for Agent Orchestration.
These can be used to describe the Axon to an LLM.
"""

from typing import Any


def get_rag_tool_definition(api_base_url: str = "http://localhost:8000") -> list[dict[str, Any]]:
    """
    Returns tool definitions compatible with OpenAI/Ollama/Anthropic tool calling.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "query_knowledge_base",
                "description": "Query the local knowledge base for a synthesized answer. Best for general questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to ask the knowledge base.",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional metadata filters (e.g. {'type': 'text'})",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Override number of chunks to retrieve for context.",
                        },
                        "project": {
                            "type": "string",
                            "description": "Target project. Must match the active project; use switch_project first if needed.",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Override similarity threshold for this request (0.0-1.0).",
                        },
                        "hybrid": {
                            "type": "boolean",
                            "description": "Override hybrid BM25+vector search toggle for this request.",
                        },
                        "rerank": {
                            "type": "boolean",
                            "description": "Override cross-encoder re-ranking toggle for this request.",
                        },
                        "hyde": {
                            "type": "boolean",
                            "description": "Override HyDE query transformation toggle for this request.",
                        },
                        "multi_query": {
                            "type": "boolean",
                            "description": "Override multi-query retrieval toggle for this request.",
                        },
                        "step_back": {
                            "type": "boolean",
                            "description": "Override step-back prompting toggle for this request.",
                        },
                        "decompose": {
                            "type": "boolean",
                            "description": "Override query decomposition toggle for this request.",
                        },
                        "compress": {
                            "type": "boolean",
                            "description": "Override LLM context compression toggle for this request.",
                        },
                        "discuss": {
                            "type": "boolean",
                            "description": "Override discussion fallback toggle for this request.",
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Override LLM temperature for this request (0.0-2.0).",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Query timeout in seconds (default 120).",
                        },
                        "include_diagnostics": {
                            "type": "boolean",
                            "description": "When true, include retrieval diagnostics in the response.",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Skip LLM generation; return ranked chunks and diagnostics only.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Retrieve raw document chunks from the knowledge base. Best for multi-step reasoning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_knowledge",
                "description": "Add new text information to the knowledge base. Useful for learning new facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The information to save."},
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata like {'source': 'web', 'topic': 'finance'}",
                        },
                        "project": {
                            "type": "string",
                            "description": "Optional project. Defaults to the active project.",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_documents",
                "description": "Remove documents from the knowledge base by their IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document IDs to delete.",
                        }
                    },
                    "required": ["doc_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ingest_directory",
                "description": "Ingest a local file or directory into the knowledge base. Returns a job_id for polling via get_job_status. Path must be within the allowed base directory (RAG_INGEST_BASE).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to a file or directory to ingest.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stream_query",
                "description": "Stream a synthesized answer token by token from the knowledge base via Server-Sent Events. Use for interactive / real-time display.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The question to ask."},
                        "filters": {"type": "object", "description": "Optional metadata filters."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_knowledge_base",
                "description": "List all unique source files ingested into the knowledge base, with chunk counts. Calls GET /collection.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_job_status",
                "description": "Poll the status of an async directory ingest job started by ingest_directory. Returns status: processing | completed | failed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The job_id returned by ingest_directory.",
                        }
                    },
                    "required": ["job_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_texts",
                "description": "Add multiple text documents to the knowledge base in a single batched embedding call. Prefer this over calling add_knowledge repeatedly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docs": {
                            "type": "array",
                            "description": "List of documents to ingest.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "The text content to ingest.",
                                    },
                                    "doc_id": {
                                        "type": "string",
                                        "description": "Optional stable ID. A UUID is assigned if omitted.",
                                    },
                                    "metadata": {
                                        "type": "object",
                                        "description": "Optional metadata dict (e.g. {'source': 'https://...', 'topic': 'react'}).",
                                    },
                                },
                                "required": ["text"],
                            },
                        },
                        "project": {
                            "type": "string",
                            "description": "Optional project applied to all docs in this batch.",
                        },
                    },
                    "required": ["docs"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ingest_url",
                "description": "Fetch an HTTP/HTTPS URL and ingest its text content into the knowledge base. HTML is stripped automatically. Private/internal URLs are blocked.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The HTTP or HTTPS URL to fetch and ingest.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional extra metadata merged with the page's source metadata.",
                        },
                        "project": {
                            "type": "string",
                            "description": "Optional project. Defaults to the active project.",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_projects",
                "description": "List all knowledge base projects known to the system, including on-disk projects and any project seen only in the current server session.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stale_docs",
                "description": "Return documents that have not been re-ingested within a given number of days. Useful for identifying knowledge that may be outdated and needs refreshing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Flag documents not re-ingested within this many days. Defaults to 7.",
                            "default": 7,
                        }
                    },
                    "required": [],
                },
            },
        },
        # ── Project CRUD ─────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "switch_project",
                "description": "Switch the active knowledge base project. Subsequent ingest/search calls use the new project. WARNING: mutates global server state — do not call from concurrent handlers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project to activate (e.g. 'react-docs', 'research/papers').",
                        }
                    },
                    "required": ["project_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_project",
                "description": "Create a new knowledge base project namespace. Projects support up to 5 slash-separated segments (e.g. 'research/papers/2024').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Project name (1-5 slash-separated alphanumeric/hyphen/underscore segments).",
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the project contents.",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_project",
                "description": "Delete a knowledge base project and all its stored documents. DANGER: irreversible. Revoke all active shares before calling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the project to delete.",
                        }
                    },
                    "required": ["name"],
                },
            },
        },
        # ── Collection management ─────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "clear_knowledge",
                "description": "Wipe all documents from the active project's vector store and BM25 index. Use to reset a project without deleting the namespace itself.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "refresh_ingest",
                "description": "Re-ingest all tracked sources whose content has changed on disk since last indexed. Returns a job_id for async polling via get_job_status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Target project. Omit to use the active project.",
                        }
                    },
                    "required": [],
                },
            },
        },
        # ── Config ────────────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "get_current_settings",
                "description": "Return the active Axon RAG configuration: top_k, similarity_threshold, hybrid_search, rerank, hyde, and all other retrieval flags.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_settings",
                "description": "Update global Axon RAG and retrieval settings for the current session. Changes are not persisted to config.yaml unless persist=true is also sent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to retrieve (1-50).",
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum match score (0.0-1.0).",
                        },
                        "hybrid_search": {
                            "type": "boolean",
                            "description": "Toggle hybrid BM25 + Vector search.",
                        },
                        "rerank": {
                            "type": "boolean",
                            "description": "Toggle cross-encoder reranking.",
                        },
                        "hyde": {
                            "type": "boolean",
                            "description": "Toggle Hypothetical Document Embeddings.",
                        },
                        "multi_query": {
                            "type": "boolean",
                            "description": "Toggle multi-query retrieval.",
                        },
                        "step_back": {
                            "type": "boolean",
                            "description": "Toggle step-back prompting.",
                        },
                        "query_decompose": {
                            "type": "boolean",
                            "description": "Toggle query decomposition.",
                        },
                        "compress_context": {
                            "type": "boolean",
                            "description": "Toggle LLM context compression.",
                        },
                        "graph_rag": {
                            "type": "boolean",
                            "description": "Toggle GraphRAG entity expansion.",
                        },
                        "raptor": {
                            "type": "boolean",
                            "description": "Toggle RAPTOR hierarchical summaries.",
                        },
                        "sentence_window": {
                            "type": "boolean",
                            "description": "Toggle sentence-window retrieval.",
                        },
                        "sentence_window_size": {
                            "type": "integer",
                            "description": "Surrounding sentences per side (1-10).",
                        },
                        "crag_lite": {
                            "type": "boolean",
                            "description": "Toggle CRAG-lite corrective retrieval.",
                        },
                        "code_graph": {
                            "type": "boolean",
                            "description": "Toggle code-graph retrieval.",
                        },
                        "graph_rag_mode": {
                            "type": "string",
                            "description": "GraphRAG mode: 'local', 'global', or 'hybrid'.",
                        },
                        "cite": {
                            "type": "boolean",
                            "description": "Include inline source citations in answers.",
                        },
                        "truth_grounding": {
                            "type": "boolean",
                            "description": "Toggle truth-grounding enforcement on retrieved chunks.",
                        },
                        "discussion_fallback": {
                            "type": "boolean",
                            "description": "Allow general-knowledge fallback when no chunks are found.",
                        },
                        "llm_provider": {
                            "type": "string",
                            "description": "LLM provider (e.g. 'ollama', 'openai', 'gemini').",
                        },
                        "llm_model": {
                            "type": "string",
                            "description": "LLM model name (e.g. 'llama3.2', 'gpt-4o').",
                        },
                        "embedding_provider": {
                            "type": "string",
                            "description": "Embedding provider (e.g. 'sentence_transformers', 'ollama').",
                        },
                        "embedding_model": {
                            "type": "string",
                            "description": "Embedding model name.",
                        },
                        "hybrid_weight": {
                            "type": "number",
                            "description": "BM25 weight in hybrid search (0.0 = pure vector, 1.0 = pure BM25).",
                        },
                        "reranker_model": {
                            "type": "string",
                            "description": "Cross-encoder reranker model name.",
                        },
                        "code_graph_bridge": {
                            "type": "boolean",
                            "description": "Toggle MENTIONED_IN edges linking prose docs to code symbols.",
                        },
                        "graph_rag_community": {
                            "type": "boolean",
                            "description": "Toggle community-summary augmentation in GraphRAG.",
                        },
                        "graph_rag_relations": {
                            "type": "boolean",
                            "description": "Toggle relation extraction in GraphRAG.",
                        },
                        "graph_rag_ner_backend": {
                            "type": "string",
                            "description": "NER backend for entity extraction: 'llm' or 'gliner'.",
                        },
                        "graph_rag_depth": {
                            "type": "string",
                            "description": "GraphRAG retrieval depth: 'light', 'standard', or 'deep'.",
                        },
                        "graph_rag_budget": {
                            "type": "integer",
                            "description": "Maximum number of entities to expand per query (>= 1).",
                        },
                        "graph_rag_relation_budget": {
                            "type": "integer",
                            "description": "Maximum number of relations to include per entity (>= 1).",
                        },
                        "persist": {
                            "type": "boolean",
                            "description": "Save these settings to config.yaml so they survive restarts. Defaults to false (session-only).",
                            "default": False,
                        },
                    },
                    "required": [],
                },
            },
        },
        # ── Sessions ──────────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "list_sessions",
                "description": "List all saved chat sessions for the active project (up to 20 most recent).",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_session",
                "description": "Retrieve a specific saved chat session transcript by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The session ID returned by list_sessions.",
                        }
                    },
                    "required": ["session_id"],
                },
            },
        },
        # ── Share / Store ─────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "get_store_status",
                "description": "Check whether the AxonStore has been initialised. Returns store metadata or {initialized: false} on a fresh install.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "init_store",
                "description": "Initialise AxonStore multi-user mode at the given base directory. Must be called once before share tools will work.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base_path": {
                            "type": "string",
                            "description": "Absolute path to the directory where AxonStore/ will be created.",
                        },
                        "persist": {
                            "type": "boolean",
                            "description": "Write the new store path to config.yaml so it survives restarts. Defaults to false.",
                            "default": False,
                        },
                    },
                    "required": ["base_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "share_project",
                "description": "Generate a read-only share key allowing another user to access one of your projects. Transmit the returned share_string out-of-band to the grantee.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Name of the project to share (must exist).",
                        },
                        "grantee": {
                            "type": "string",
                            "description": "OS username of the recipient.",
                        },
                    },
                    "required": ["project", "grantee"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "redeem_share",
                "description": "Redeem a share string to mount a shared project in your mounts/ directory. The shared project can then be queried normally.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "share_string": {
                            "type": "string",
                            "description": "The base64 share string generated by share_project on the owner's machine.",
                        }
                    },
                    "required": ["share_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "revoke_share",
                "description": "Revoke a previously generated share key, immediately cutting off the grantee's access.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key_id": {
                            "type": "string",
                            "description": "The key_id of the share to revoke (from list_shares output).",
                        }
                    },
                    "required": ["key_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_shares",
                "description": "List all active shares: projects shared by you (sharing) and projects shared with you (shared). Use to audit access or troubleshoot missing shared projects.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        # ── Graph ─────────────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "graph_status",
                "description": "Return GraphRAG knowledge-graph status: entity count, edge count, community summary count, and whether a rebuild is in progress.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_finalize",
                "description": "Trigger an explicit GraphRAG community detection rebuild. Call after a large ingest batch to update graph-augmented retrieval without waiting for the automatic rebuild.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_data",
                "description": "Return the full entity/relation knowledge-graph as a JSON nodes+links payload. Useful for inspection, export, or building custom visualisations.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_backend_status",
                "description": "Return the active graph backend's health metrics: which backend is active (graphrag or dynamic), whether it is ready, entity count, edge count, etc.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        # ── Governance ────────────────────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "get_active_leases",
                "description": "Return active write-lease counts for all projects tracked by the server. Use to check whether it is safe to put a project into maintenance state.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


# Implementation of tool logic for a simple agent framework could go here
# or the user can just call the API.
