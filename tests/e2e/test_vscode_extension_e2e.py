from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_vscode_extension_manifest_exposes_graph_commands_and_tool():
    manifest_path = Path("integrations/vscode-axon/package.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    activation_events = set(manifest.get("activationEvents", []))

    commands = {item["command"] for item in manifest["contributes"]["commands"]}

    tool_names = {item["name"] for item in manifest["contributes"]["languageModelTools"]}

    assert "onLanguageModelTool:show_graph" in activation_events

    assert "axon.showGraphForQuery" in commands

    assert "axon.showGraphForSelection" in commands

    assert "show_graph" in tool_names


def test_vscode_extension_activation_and_graph_panel_smoke(make_brain, live_api_server, tmp_path):
    import shutil

    if not shutil.which("node"):
        pytest.skip("Node.js not available")
    ext_js = Path("integrations/vscode-axon/out/extension.js")
    if not ext_js.exists():
        pytest.skip("Extension not built — run: cd integrations/vscode-axon && npm run compile")

    brain = make_brain(graph_rag=True, code_graph=True)

    graph_doc = tmp_path / "graph_note.txt"

    graph_doc.write_text(
        "GraphRAG connects entities to support graph-aware retrieval and synthesis.",
        encoding="utf-8",
    )

    brain.ingest(
        [
            {
                "id": "graph_doc",
                "text": "GraphRAG connects entities to support graph-aware retrieval and synthesis.",
                "metadata": {"source": str(graph_doc), "start_line": 12},
            }
        ]
    )

    brain._entity_graph = {
        "graphrag": {
            "type": "CONCEPT",
            "description": "Graph-aware retrieval layer",
            "chunk_ids": ["graph_doc"],
            "degree": 1,
        },
        "retrieval": {
            "type": "CONCEPT",
            "description": "Retrieval pipeline",
            "chunk_ids": ["graph_doc"],
            "degree": 1,
        },
    }

    brain._relation_graph = {
        "graphrag": [
            {
                "target": "retrieval",
                "relation": "supports",
                "description": "GraphRAG supports retrieval",
                "weight": 3,
            }
        ]
    }

    brain._community_levels = {0: {"graphrag": 1, "retrieval": 1}}

    brain._code_graph = {
        "nodes": {
            "file::extension": {
                "name": "extension.ts",
                "node_type": "file",
                "file_path": str(Path("integrations/vscode-axon/src/extension.ts").resolve()),
                "start_line": 1,
                "chunk_ids": ["graph_doc"],
            },
            "fn::showGraphForQuery": {
                "name": "showGraphForQuery",
                "node_type": "function",
                "signature": "showGraphForQuery(context, query)",
                "file_path": str(Path("integrations/vscode-axon/src/extension.ts").resolve()),
                "start_line": 185,
                "chunk_ids": ["graph_doc"],
            },
        },
        "edges": [
            {
                "source": "file::extension",
                "target": "fn::showGraphForQuery",
                "edge_type": "CONTAINS",
            }
        ],
    }

    base_url = live_api_server()

    runner = tmp_path / "vscode_extension_runner.js"

    runner.write_text(
        textwrap.dedent(
            f"""


            const path = require('path');


            const Module = require('module');


            const extensionPath = process.argv[2];


            const extensionRoot = process.argv[3];


            const apiBase = process.argv[4];


            const queryText = process.argv[5];


            const commands = new Map();


            const toolNames = [];


            const openedDocs = [];


            const errors = [];


            const infos = [];


            const warnings = [];


            const panels = [];


            const vscode = {{


              ViewColumn: {{ Beside: 2 }},


              ThemeIcon: class ThemeIcon {{ constructor(id) {{ this.id = id; }} }},


              Range: class Range {{ constructor(sLine, sChar, eLine, eChar) {{ this.start = {{ line: sLine, character: sChar }}; this.end = {{ line: eLine, character: eChar }}; }} }},


              Uri: {{


                joinPath(base, ...parts) {{ return {{ fsPath: path.join(base.fsPath, ...parts) }}; }},


                file(fp) {{ return {{ fsPath: fp }}; }},


              }},


              window: {{


                createOutputChannel() {{ return {{ appendLine() {{}}, dispose() {{}} }}; }},


                createWebviewPanel(viewType, title, column, options) {{


                  const panel = {{


                    viewType,


                    title,


                    column,


                    options,


                    reveal() {{}},


                    dispose() {{ this.disposed = true; if (this._disposeCb) this._disposeCb(); }},


                    onDidDispose(cb) {{ this._disposeCb = cb; return {{ dispose() {{}} }}; }},


                    webview: {{


                      html: '',


                      cspSource: 'vscode-webview-resource:',


                      asWebviewUri(uri) {{ return `file://${{uri.fsPath.replace(/\\\\/g, '/')}}`; }},


                      onDidReceiveMessage(cb) {{ panel._messageCb = cb; return {{ dispose() {{}} }}; }},


                    }},


                  }};


                  panels.push(panel);


                  return panel;


                }},


                showInputBox() {{ return Promise.resolve(queryText); }},


                showErrorMessage(msg) {{ errors.push(String(msg)); return Promise.resolve(); }},


                showInformationMessage(msg) {{ infos.push(String(msg)); return Promise.resolve(); }},


                showWarningMessage(msg) {{ warnings.push(String(msg)); return Promise.resolve(); }},


                showTextDocument(uri, opts) {{ openedDocs.push({{ path: uri.fsPath, line: opts?.selection?.start?.line ?? 0 }}); return Promise.resolve(); }},


                activeTextEditor: {{ document: {{ getText() {{ return queryText; }} }}, selection: {{}} }},


              }},


              workspace: {{


                getConfiguration() {{


                  return {{


                    get(key, fallback) {{


                      const values = {{


                        apiBase,


                        apiKey: '',


                        autoStart: false,


                        useCopilotLlm: false,


                      }};


                      return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : fallback;


                    }}


                  }};


                }},


              }},


              commands: {{


                registerCommand(name, fn) {{ commands.set(name, fn); return {{ dispose() {{ commands.delete(name); }} }}; }},


                async executeCommand(name, ...args) {{ return commands.get(name)(...args); }},


              }},


              chat: {{


                createChatParticipant() {{ return {{ iconPath: undefined, dispose() {{}} }}; }},


              }},


              lm: {{


                registerTool(name) {{ toolNames.push(name); return {{ dispose() {{}} }}; }},


              }},


            }};


            const originalLoad = Module._load;


            Module._load = function(request, parent, isMain) {{


              if (request === 'vscode') {{


                return vscode;


              }}


              return originalLoad.apply(this, arguments);


            }};


            async function run() {{


              const extension = require(extensionPath);


              const context = {{


                extensionUri: {{ fsPath: extensionRoot }},


                subscriptions: [],


              }};


              await extension.activate(context);


              await commands.get('axon.showGraphForQuery')();


              const panel = panels[0];


              if (!panel) {{


                throw new Error('Graph panel was not created');


              }}


              if (panel._messageCb) {{


                panel._messageCb({{ command: 'openFile', path: '{Path("integrations/vscode-axon/src/extension.ts").resolve().as_posix()}', line: 25 }});


              }}


              process.stdout.write(JSON.stringify({{


                commands: Array.from(commands.keys()).sort(),


                toolNames: toolNames.sort(),


                html: panel.webview.html,


                openedDocs,


                errors,


                infos,


                warnings,


              }}));


            }}


            run().catch((err) => {{


              console.error(err);


              process.exit(1);


            }});


            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "node",
            str(runner),
            str(Path("integrations/vscode-axon/out/extension.js").resolve()),
            str(Path("integrations/vscode-axon").resolve()),
            base_url,
            "How does GraphRAG help retrieval?",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path.cwd(),
    )

    payload = json.loads(result.stdout)

    assert "axon.showGraphForQuery" in payload["commands"]

    assert "axon.showGraphForSelection" in payload["commands"]

    assert "show_graph" in payload["toolNames"]

    assert "graphrag" in payload["html"].lower()

    assert "Knowledge Graph" in payload["html"]

    assert "Code Graph" in payload["html"]

    assert "graph-panel.js" in payload["html"]

    assert "script-src vscode-webview-resource:" in payload["html"]

    assert payload["openedDocs"]

    opened_path = payload["openedDocs"][0]["path"].replace("\\", "/")

    assert opened_path.endswith("integrations/vscode-axon/src/extension.ts")

    assert payload["errors"] == []
