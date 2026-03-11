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
                "description": "Ingest a local file or directory into the knowledge base. Triggers background processing. Path must be within the allowed base directory (RAG_INGEST_BASE).",
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
                        }
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
                    },
                    "required": ["url"],
                },
            },
        },
    ]


# Implementation of tool logic for a simple agent framework could go here
# or the user can just call the API.
