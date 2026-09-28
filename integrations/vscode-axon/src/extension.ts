/**
 * Axon VS Code extension — activation / deactivation root.
 *
 * All business logic lives in sub-modules:
 *   shared.ts          — mutable extension state
 *   client/http.ts     — HTTP helpers
 *   client/server.ts   — server lifecycle
 *   graph/panel.ts     — AxonGraphPanel webview
 *   tools/query.ts     — search / query LM tools + chat participant
 *   tools/ingest.ts    — ingest LM tools + commands
 *   tools/projects.ts  — project management LM tools + commands
 *   tools/shares.ts    — share LM tools + commands
 *   tools/graph.ts     — graph LM tools + commands
 *   tools/config.ts    — config LM tools + setup wizard command
 */

import * as vscode from 'vscode';

import { state, resolveApiBase } from './shared';

import { ensureServerRunning, stopServer, waitForHealth } from './client/server';

import { httpPost } from './client/http';

import { showGraphForQuery, showGraphForSelection } from './graph/panel';

import { makeChatHandler, AxonSearchTool, AxonQueryTool } from './tools/query';

import {
  AxonIngestKnowledgeTool, AxonGetIngestStatusTool, AxonIngestImageTool,
  ingestCurrentFile, ingestWorkspaceFolder, ingestPickedFolder,
  refreshIngest, listStaleDocs, clearKnowledgeBase,

} from './tools/ingest';

import {
  AxonListProjectsTool, AxonSwitchProjectTool, AxonCreateProjectTool,
  AxonDeleteDocumentsTool, AxonListKnowledgeTool,
  switchProject, createNewProject,

} from './tools/projects';

import {
  AxonShareProjectTool, AxonRedeemShareTool, AxonRevokeShareTool,
  AxonListSharesTool, AxonExtendShareTool,
  initStore, shareProject, redeemShare, revokeShare, listShares,

} from './tools/shares';

import {
  AxonShowGraphTool, AxonGraphRetrieveTool, AxonUpdateFactTool,
  showGraphStatus,

} from './tools/graph';

import { AxonConfigGetTool, AxonConfigSetTool, runConfigSetupWizard } from './tools/config';

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  state.outputChannel = vscode.window.createOutputChannel('Axon');
  context.subscriptions.push(state.outputChannel);
  const config = vscode.workspace.getConfiguration('axon');
  const autoStart = config.get<boolean>('autoStart', true);
  state.outputChannel.appendLine('Axon extension activating.');
  if (autoStart) {
    await ensureServerRunning(context);
  }
  // Resolved after ensureServerRunning (which sets state.apiBase); commands
  // registered below read it fresh at call time via resolveApiBase() rather
  // than closing over this snapshot, since the address can change between
  // activation and any individual command invocation (e.g. autoStart off,
  // server started later via axon.startServer).
  const apiBase = resolveApiBase();
  const useCopilotLlm = config.get<boolean>('useCopilotLlm', false);
  const apiKey = config.get<string>('apiKey', '');
  if (useCopilotLlm) {
    state.outputChannel.appendLine('Axon: Using Copilot LLM for backend tasks.');
    // Import lazily to avoid circular-import issues with server.ts
    const { startCopilotLlmWorker } = await import('./client/server');
    startCopilotLlmWorker(apiKey);
    // Tell the backend to use the 'copilot' provider and PERSIST it
    waitForHealth(apiBase, 120_000).then((running) => {
      if (running) {
        httpPost(`${apiBase}/config/update`, { llm_provider: 'copilot', persist: true }, apiKey)
          .then(() => state.outputChannel.appendLine('Axon backend configured to use Copilot provider (persistent).'))
          .catch((err) => state.outputChannel.appendLine(`Failed to set copilot provider: ${err}`));
      }
    });
  }
  // Register the @axon chat participant
  const participant = vscode.chat.createChatParticipant('axon.chat', makeChatHandler(context));
  participant.iconPath = new vscode.ThemeIcon('database');
  context.subscriptions.push(participant);
  // Register commands. Each reads resolveApiBase() fresh at invocation time
  // (not a closure-captured snapshot) since the server address can change
  // between activation and any individual command invocation.
  context.subscriptions.push(
    vscode.commands.registerCommand('axon.switchProject', () => switchProject(resolveApiBase())),
    vscode.commands.registerCommand('axon.createProject', () => createNewProject(resolveApiBase())),
    vscode.commands.registerCommand('axon.ingestFile', () => ingestCurrentFile(resolveApiBase())),
    vscode.commands.registerCommand('axon.ingestWorkspace', () => ingestWorkspaceFolder(resolveApiBase())),
    vscode.commands.registerCommand('axon.ingestFolder', () => ingestPickedFolder(resolveApiBase())),
    vscode.commands.registerCommand('axon.startServer', () => ensureServerRunning(context)),
    vscode.commands.registerCommand('axon.stopServer', () => stopServer()),
    vscode.commands.registerCommand('axon.initStore', () => initStore(resolveApiBase())),
    vscode.commands.registerCommand('axon.shareProject', () => shareProject(resolveApiBase())),
    vscode.commands.registerCommand('axon.redeemShare', () => redeemShare(resolveApiBase())),
    vscode.commands.registerCommand('axon.revokeShare', () => revokeShare(resolveApiBase())),
    vscode.commands.registerCommand('axon.listShares', () => listShares(resolveApiBase())),
    vscode.commands.registerCommand('axon.refreshIngest', () => refreshIngest(resolveApiBase())),
    vscode.commands.registerCommand('axon.listStaleDocs', () => listStaleDocs(resolveApiBase())),
    vscode.commands.registerCommand('axon.clearKnowledgeBase', () => clearKnowledgeBase(resolveApiBase())),
    vscode.commands.registerCommand('axon.showGraphStatus', () => showGraphStatus(resolveApiBase())),
    vscode.commands.registerCommand('axon.showGraphForQuery', async () => {
      const query = await vscode.window.showInputBox({ prompt: 'Axon: Enter query to visualise' });
      if (query) { await showGraphForQuery(context, query); }
    }),
    vscode.commands.registerCommand('axon.showGraphForSelection', () => showGraphForSelection(context)),
    vscode.commands.registerCommand('axon.configSetup', async () => {
      const cfg = vscode.workspace.getConfiguration('axon');
      const apiKey = cfg.get<string>('apiKey', '');
      await runConfigSetupWizard(resolveApiBase(), apiKey);
    }),
  );
  // Register Language Model Tools (for the Copilot agent toolset). Exactly the
  // tools declared in package.json contributes.languageModelTools — the 18 MCP
  // tools plus show_graph and ingest_image (0.5.0). Destructive and admin
  // operations stay human-only: they are the commands registered above.
  try {
    if ('lm' in vscode && (vscode as any).lm.registerTool) {
      state.outputChannel.appendLine('Registering Axon Language Model Tools...');
      context.subscriptions.push(
        (vscode as any).lm.registerTool('search_knowledge', new AxonSearchTool(context)),
        (vscode as any).lm.registerTool('query_knowledge', new AxonQueryTool(context)),
        (vscode as any).lm.registerTool('ingest_knowledge', new AxonIngestKnowledgeTool()),
        (vscode as any).lm.registerTool('get_job_status', new AxonGetIngestStatusTool()),
        (vscode as any).lm.registerTool('list_knowledge', new AxonListKnowledgeTool()),
        (vscode as any).lm.registerTool('delete_documents', new AxonDeleteDocumentsTool()),
        (vscode as any).lm.registerTool('list_projects', new AxonListProjectsTool()),
        (vscode as any).lm.registerTool('switch_project', new AxonSwitchProjectTool()),
        (vscode as any).lm.registerTool('create_project', new AxonCreateProjectTool()),
        (vscode as any).lm.registerTool('get_config', new AxonConfigGetTool()),
        (vscode as any).lm.registerTool('set_config', new AxonConfigSetTool()),
        (vscode as any).lm.registerTool('graph_retrieve', new AxonGraphRetrieveTool()),
        (vscode as any).lm.registerTool('update_fact', new AxonUpdateFactTool()),
        (vscode as any).lm.registerTool('share_project', new AxonShareProjectTool()),
        (vscode as any).lm.registerTool('redeem_share', new AxonRedeemShareTool()),
        (vscode as any).lm.registerTool('list_shares', new AxonListSharesTool()),
        (vscode as any).lm.registerTool('revoke_share', new AxonRevokeShareTool()),
        (vscode as any).lm.registerTool('extend_share', new AxonExtendShareTool()),
        (vscode as any).lm.registerTool('show_graph', new AxonShowGraphTool(context)),
        (vscode as any).lm.registerTool('ingest_image', new AxonIngestImageTool()),
      );
      state.outputChannel.appendLine('Successfully registered all Axon tools.');
    } else {
      state.outputChannel.appendLine('Language Model Tools API not available in this VS Code version.');
    }
  } catch (err) {
    state.outputChannel.appendLine(`Error registering tools: ${err}`);
  }
  state.outputChannel.appendLine('Axon extension ready.');

}

export function deactivate(): void {
  stopServer();

}

