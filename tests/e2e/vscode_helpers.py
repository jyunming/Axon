"""Shared helpers for VS Code extension e2e qualification tests.

Provides:
- _RUNNER_JS: Node.js template that stubs the ``vscode`` module,
  activates the extension, and optionally invokes one named LM tool.
- run_extension_tool(): runs the Node.js subprocess and returns a
  result dict with ``toolResult``, ``registeredTools``, etc.
- mock_api_response(): returns a realistic mock JSON payload for a
  given API path — used by ``live_recorder_server`` in conftest.py.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Node.js runner template
# ---------------------------------------------------------------------------

_RUNNER_JS = r"""
'use strict';
// Axon VS Code Extension Test Runner
// Args: extensionJs extensionRoot apiBase toolName toolInputJSON extraConfigJSON

const path = require('path');
const Module = require('module');

const [extensionJs, extensionRoot, apiBase, toolName, toolInputRaw, extraConfigRaw] =
  process.argv.slice(2);

const toolInput   = JSON.parse(toolInputRaw   || '{}');
const extra       = JSON.parse(extraConfigRaw || '{}');

const registeredTools    = {};
const registeredCommands = new Map();
const registeredParticipants = [];
const outputLines    = [];
const openedDocs     = [];
const clipboardWrites = [];
let   panelCount     = 0;
let   lastPanelHtml  = '';
let   inputResponseIndex = 0;

// -- Minimal LanguageModel stubs -------------------------------------------
class LanguageModelTextPart {
  constructor(v) { this.value = v; }
}
class LanguageModelDataPart {
  constructor(data, mime) { this.data = data; this.mimeType = mime; }
}
class LanguageModelToolResult {
  constructor(parts) { this.parts = parts; }
}

// -- vscode stub --------------------------------------------------------------
const vscode = {
  version: '1.96.0',
  ViewColumn: { Beside: 2, One: 1, Active: -1 },
  ThemeIcon:  class ThemeIcon { constructor(id) { this.id = id; } },
  Range: class Range {
    constructor(sl, sc, el, ec) {
      this.start = { line: sl, character: sc };
      this.end   = { line: el, character: ec };
    }
  },
  Uri: {
    file: (p) => ({ fsPath: p, toString: () => 'file://' + p }),
    joinPath: (base, ...parts) => {
      const b = (base && base.fsPath) ? base.fsPath : (typeof base === 'string' ? base : '');
      const joined = path.join(b, ...parts);
      return { fsPath: joined, toString: () => 'file://' + joined };
    },
  },
  LanguageModelChatMessage: {
    User: (parts) => ({ role: 'user', content: parts }),
  },
  LanguageModelTextPart,
  LanguageModelDataPart,
  LanguageModelToolResult,

  window: {
    createOutputChannel: () => ({
      appendLine: (l) => outputLines.push(l),
      show: () => {},
      dispose: () => {},
    }),
    createWebviewPanel: (viewType, title, column, options) => {
      panelCount += 1;
      let html = '';
      const panel = {
        viewType, title, column, options,
        webview: {
          cspSource: 'vscode-webview-resource:',
          asWebviewUri: (uri) => uri,
          onDidReceiveMessage: (cb) => { panel._messageCb = cb; return { dispose() {} }; },
          postMessage: () => {},
        },
        reveal: () => {},
        dispose: () => {},
        onDidDispose: () => ({ dispose() {} }),
      };
      Object.defineProperty(panel.webview, 'html', {
        get: () => html,
        set: (value) => {
          html = value;
          lastPanelHtml = value;
        },
      });
      return panel;
    },
    showInputBox:          async ()     => {
      if (Array.isArray(extra._inputResponses) && inputResponseIndex < extra._inputResponses.length) {
        const value = extra._inputResponses[inputResponseIndex];
        inputResponseIndex += 1;
        return value;
      }
      return extra._inputResponse !== undefined ? extra._inputResponse : null;
    },
    showQuickPick:         async (items) => {
      if (extra._quickPickResponse !== undefined) { return extra._quickPickResponse; }
      if (!items || items.length === 0) { return null; }
      return items[0];
    },
    showErrorMessage:      async (msg)  => { outputLines.push('[ERR] '  + msg); return undefined; },
    showInformationMessage:async (msg)  => { outputLines.push('[INFO] ' + msg); return undefined; },
    showWarningMessage:    async (msg, ...btns) => {
      outputLines.push('[WARN] ' + msg);
      return extra._confirmResponse !== undefined ? extra._confirmResponse : (btns[0] || undefined);
    },
    showTextDocument: async (uri, opts) => {
      openedDocs.push({
        path: (uri && uri.fsPath) ? uri.fsPath : String(uri),
        line: (opts && opts.selection) ? opts.selection.start.line : 0,
      });
    },
    activeTextEditor: {
      document: {
        uri: { fsPath: extra._activeFile || path.join(extensionRoot, 'sample.txt') },
        getText: () => (extra._selectedText || extra._activeFileText || 'sample file content'),
      },
      selection: { isEmpty: false },
    },
    showOpenDialog: async () =>
      extra._openDialogResult ? [{ fsPath: extra._openDialogResult }] : undefined,
  },

  workspace: {
    getConfiguration: (_section) => ({
      get: (key, fallback) => {
        const vals = {
          apiBase,
          apiKey:          '',
          autoStart:       false,
          useCopilotLlm:   false,
          topK:            5,
          storeBase:       '',
          ingestBase:      '',
          pythonPath:      '',
        };
        Object.assign(vals, extra);
        return Object.prototype.hasOwnProperty.call(vals, key) ? vals[key] : fallback;
      },
    }),
    workspaceFolders: (extra._workspaceFolders || []).map(p => ({ uri: { fsPath: p } })),
    openTextDocument:  async (uri) => ({ uri, getText: () => '' }),
    findFiles:         async ()    => [],
  },

  commands: {
    registerCommand: (name, fn) => {
      registeredCommands.set(name, fn);
      return { dispose() { registeredCommands.delete(name); } };
    },
    executeCommand: async (name, ...args) => {
      const fn = registeredCommands.get(name);
      if (fn) { return fn(...args); }
    },
  },

  chat: {
    createChatParticipant: (id, handler) => {
      registeredParticipants.push({ id, handler });
      return { iconPath: undefined, dispose() {} };
    },
  },

  lm: {
    registerTool: (name, tool) => {
      registeredTools[name] = tool;
      return { dispose() {} };
    },
    selectChatModels: async () => {
      const rawModels = extra._copilotModels || [{
        id: 'gpt-4o',
        capabilities: { supportsImageToText: true },
      }];
      return rawModels.map((model) => ({
        ...model,
        sendRequest: model.sendRequest || (async () => ({
          text: (async function* () {
            yield model._responseText || extra._copilotResponseText || 'Mock image description';
          })(),
        })),
      }));
    },
  },

  env: {
    clipboard: {
      writeText: async (text) => { clipboardWrites.push(text); },
    },
  },
};

// -- Intercept require('vscode') -----------------------------------------------
const _origLoad = Module._load.bind(Module);
Module._load = function(request, parent, isMain) {
  if (request === 'vscode') { return vscode; }
  return _origLoad(request, parent, isMain);
};

// -- Main runner ---------------------------------------------------------------
async function main() {
  const ext     = require(extensionJs);
  const context = { extensionUri: { fsPath: extensionRoot }, subscriptions: [] };
  await ext.activate(context);

  const postActivateWaitMs = extra._postActivateWaitMs || 0;
  if (postActivateWaitMs > 0) {
    await new Promise((resolve) => setTimeout(resolve, postActivateWaitMs));
  }

  let toolResult = null;
  let toolError  = null;

  if (toolName) {
    if (toolName.startsWith('cmd:')) {
      const commandName = toolName.slice(4);
      try {
        await vscode.commands.executeCommand(commandName);
        toolResult = 'Command executed: ' + commandName;
      } catch (err) {
        toolError = String(err);
      }
    } else {
      const tool = registeredTools[toolName];
      if (!tool) {
        toolError = 'Tool not registered: ' + toolName +
                    '. Available: ' + Object.keys(registeredTools).join(', ');
      } else {
        try {
          const result = await tool.invoke({ input: toolInput }, {});
          if (result && result.parts) {
            toolResult = result.parts.map(p => (p.value || '')).join('');
          } else {
            toolResult = String(result);
          }
        } catch (err) {
          toolError = String(err);
        }
      }
    }
  }

  process.stdout.write(JSON.stringify({
    toolResult,
    toolError,
    registeredTools:      Object.keys(registeredTools).sort(),
    registeredCommands:   Array.from(registeredCommands.keys()).sort(),
    registeredParticipants: registeredParticipants.map(p => p.id),
    outputLines,
    openedDocs,
    clipboardWrites,
    panelCount,
    lastPanelHtml,
  }));
  process.exit(0);
}

main().catch(err => {
  process.stderr.write(String(err) + '\n');
  process.exit(1);
});
"""


# ---------------------------------------------------------------------------
# Python runner helper
# ---------------------------------------------------------------------------


def run_extension_tool(
    runner_js_path: Path,
    extension_js: Path,
    extension_root: Path,
    api_base: str,
    tool_name: str,
    tool_input: dict[str, Any],
    extra_config: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Run the Node.js extension runner, invoke one LM tool, return the result dict.

    Raises ``subprocess.CalledProcessError`` on non-zero exit.
    """
    result = subprocess.run(
        [
            "node",
            str(runner_js_path),
            str(extension_js),
            str(extension_root),
            api_base,
            tool_name,
            json.dumps(tool_input),
            json.dumps(extra_config or {}),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path.cwd(),
        timeout=timeout,
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Mock API response generator (used by live_recorder_server fixture)
# ---------------------------------------------------------------------------


def mock_api_response(url_path: str, method: str, body: dict) -> dict:
    """Return a realistic mock JSON body for the given Axon API path."""
    p = url_path.rstrip("/")
    if p in ("/health", ""):
        return {"status": "ok", "project": "default", "version": "0.9.0"}
    if p == "/search":
        return {
            "results": [
                {
                    "id": "t1",
                    "text": "Axon grounded result",
                    "score": 0.91,
                    "metadata": {"source": "test.md"},
                }
            ]
        }
    if p == "/query":
        return {"response": "Axon synthesised answer.", "status": "ok"}
    if p == "/add_text":
        return {"status": "ok", "doc_id": "mock-doc-001"}
    if p == "/add_texts":
        return {"status": "ok", "doc_ids": ["mock-doc-001"]}
    if p == "/ingest_url":
        return {"status": "ok", "url": body.get("url", ""), "chunks": 3}
    if p == "/ingest":
        return {"status": "ok", "message": "Ingestion started", "job_id": "mock-job-001"}
    if p == "/ingest/refresh":
        return {
            "reingested": ["overview.md", "notes.md"],
            "skipped": ["stable.md"],
            "missing": [],
            "errors": [],
        }
    if p.startswith("/ingest/status/"):
        return {"status": "completed", "chunks": 5}
    if p == "/projects":
        return {
            "projects": [
                {"name": "default"},
                {"name": "engineering"},
                {"name": "research"},
                {"name": "team/subproject"},
            ]
        }
    if p == "/project/switch":
        return {"active_project": body.get("name", "default"), "status": "ok"}
    if p == "/project/new":
        return {"status": "ok", "project": body.get("name", "new-proj")}
    if p.startswith("/project/delete/"):
        return {"status": "ok", "message": "Project deleted"}
    if p == "/delete":
        return {"status": "ok", "deleted": len(body.get("doc_ids", []))}
    if p == "/collection":
        return {
            "total_files": 2,
            "total_chunks": 10,
            "files": [{"source": "overview.md", "chunks": 6}, {"source": "notes.md", "chunks": 4}],
        }
    if p == "/config":
        return {
            "llm_provider": "ollama",
            "llm_model": "gemma:2b",
            "top_k": 10,
            "rerank": False,
            "hybrid_search": True,
            "sentence_window": False,
            "crag_lite": False,
            "graph_rag_mode": "local",
            "cite": True,
        }
    if p == "/clear":
        return {"status": "ok", "message": "Collection cleared"}
    if p == "/config/update":
        return {"status": "ok", "message": "Settings applied"}
    if p == "/share/generate":
        return {
            "project": body.get("project", "default"),
            "grantee": body.get("grantee", "bob"),
            "key_id": "sk_test123abc",
            "share_string": "axon-share-base64placeholder",
        }
    if p == "/share/redeem":
        return {"owner": "alice", "project": "default", "mount_name": "mounts/alice_default"}
    if p == "/share/revoke":
        return {
            "key_id": body.get("key_id", "sk_test123abc"),
            "grantee": "bob",
            "project": "default",
        }
    if p == "/share/list":
        return {
            "sharing": [
                {
                    "key_id": "sk_test123abc",
                    "project": "default",
                    "grantee": "bob",
                    "revoked": False,
                }
            ],
            "shared": [],
        }
    if p == "/store/init":
        return {
            "status": "ok",
            "store_path": "/tmp/AxonStore/testuser",
            "username": "testuser",
            "user_dir": "/tmp/AxonStore/testuser/user",
        }
    if p == "/graph/status":
        return {
            "entity_count": 5,
            "relation_count": 3,
            "community_summary_count": 2,
            "community_build_in_progress": False,
        }
    if p == "/collection/stale":
        return {"stale_docs": [{"project": "default", "doc_id": "old-doc-001", "age_days": 14}]}
    if p == "/search/raw":
        return {
            "results": [
                {
                    "id": "raw1",
                    "text": "Graph source snippet",
                    "score": 0.87,
                    "metadata": {"source": "graph.md"},
                }
            ]
        }
    if p == "/graph/data":
        return {"nodes": [{"id": "n1", "label": "Entity"}], "links": []}
    if p == "/code-graph/data":
        return {"nodes": [{"id": "c1", "label": "Function"}], "links": []}
    # Default fallback
    return {"status": "ok"}
