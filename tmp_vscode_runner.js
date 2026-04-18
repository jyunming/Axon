
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
const createdPanels  = [];

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
      createdPanels.push(panel);
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
    } else if (toolName.startsWith('chat:')) {
      // Invoke a registered chat participant handler directly.
      // Usage: tool_name = "chat:axon.chat", tool_input = { prompt: "..." }
      const participantId = toolName.slice(5);
      const participant = registeredParticipants.find(p => p.id === participantId);
      if (!participant) {
        toolError = 'Chat participant not registered: ' + participantId +
                    '. Available: ' + registeredParticipants.map(p => p.id).join(', ');
      } else {
        const chatMarkdownParts = [];
        const fakeResponse = {
          markdown: (text) => { chatMarkdownParts.push(text); },
          progress: () => {},
          reference: () => {},
          anchor: () => {},
          button: () => {},
          filetree: () => {},
          push: () => {},
        };
        const fakeRequest = {
          prompt: toolInput.prompt || '',
          command: toolInput.command || undefined,
          references: [],
        };
        const fakeToken = { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) };
        try {
          await participant.handler(fakeRequest, {}, fakeResponse, fakeToken);
          toolResult = chatMarkdownParts.join('');
        } catch (err) {
          toolError = String(err);
        }
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

  const postCommandWaitMs = extra._postCommandWaitMs || 0;
  if (postCommandWaitMs > 0) {
    await new Promise((resolve) => setTimeout(resolve, postCommandWaitMs));
  }

  // Dispatch fake webview → extension messages if requested
  if (Array.isArray(extra._panelMessages) && createdPanels.length > 0) {
    const targetPanel = createdPanels[0];
    for (const msg of extra._panelMessages) {
      if (targetPanel._messageCb) {
        await targetPanel._messageCb(msg);
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
    allPanelHtmls: createdPanels.map(p => p.webview.html || ''),
  }));
  process.exit(0);
}

main().catch(err => {
  process.stderr.write((err && err.stack) ? (err.stack + '\n') : (String(err) + '\n'));
  process.exit(1);
});
