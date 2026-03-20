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
 */
import * as vscode from 'vscode';

import { state } from './shared';
import { ensureServerRunning, stopServer, waitForHealth } from './client/server';
import { httpPost } from './client/http';
import { showGraphForQuery, showGraphForSelection } from './graph/panel';
import { chatHandler, AxonSearchTool, AxonQueryTool } from './tools/query';
import {
  AxonIngestTextTool, AxonIngestUrlTool, AxonIngestPathTool,
  AxonGetIngestStatusTool, AxonIngestImageTool, AxonRefreshIngestTool,
  AxonListStaleDocsTool, AxonClearKnowledgeBaseTool,
  ingestCurrentFile, ingestWorkspaceFolder, ingestPickedFolder,
  refreshIngest, listStaleDocs, clearKnowledgeBase,
} from './tools/ingest';
import {
  AxonListProjectsTool, AxonSwitchProjectTool, AxonCreateProjectTool,
  AxonDeleteProjectTool, AxonDeleteDocumentsTool, AxonGetCollectionTool,
  AxonClearCollectionTool, AxonUpdateSettingsTool,
  switchProject, createNewProject,
} from './tools/projects';
import {
  AxonShareProjectTool, AxonRedeemShareTool, AxonRevokeShareTool,
  AxonListSharesTool, AxonInitStoreTool,
  initStore, shareProject, redeemShare, revokeShare, listShares,
} from './tools/shares';
import {
  AxonShowGraphStatusTool, AxonShowGraphTool,
  showGraphStatus,
} from './tools/graph';

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  state.outputChannel = vscode.window.createOutputChannel('Axon');
  context.subscriptions.push(state.outputChannel);

  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const autoStart = config.get<boolean>('autoStart', true);

  state.outputChannel.appendLine(`Axon extension activating. API base: ${apiBase}`);

  if (autoStart) {
    await ensureServerRunning(apiBase, context);
  }

  const useCopilotLlm = config.get<boolean>('useCopilotLlm', false);
  const apiKey = config.get<string>('apiKey', '');
  if (useCopilotLlm) {
    state.outputChannel.appendLine('Axon: Using Copilot LLM for backend tasks.');
    // Import lazily to avoid circular-import issues with server.ts
    const { startCopilotLlmWorker } = await import('./client/server');
    startCopilotLlmWorker(apiBase, apiKey);
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
  const participant = vscode.chat.createChatParticipant('axon.chat', chatHandler);
  participant.iconPath = new vscode.ThemeIcon('database');
  context.subscriptions.push(participant);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('axon.switchProject', () => switchProject(apiBase)),
    vscode.commands.registerCommand('axon.createProject', () => createNewProject(apiBase)),
    vscode.commands.registerCommand('axon.ingestFile', () => ingestCurrentFile(apiBase)),
    vscode.commands.registerCommand('axon.ingestWorkspace', () => ingestWorkspaceFolder(apiBase)),
    vscode.commands.registerCommand('axon.ingestFolder', () => ingestPickedFolder(apiBase)),
    vscode.commands.registerCommand('axon.startServer', () => ensureServerRunning(apiBase, context)),
    vscode.commands.registerCommand('axon.stopServer', () => stopServer()),
    vscode.commands.registerCommand('axon.initStore', () => initStore(apiBase)),
    vscode.commands.registerCommand('axon.shareProject', () => shareProject(apiBase)),
    vscode.commands.registerCommand('axon.redeemShare', () => redeemShare(apiBase)),
    vscode.commands.registerCommand('axon.revokeShare', () => revokeShare(apiBase)),
    vscode.commands.registerCommand('axon.listShares', () => listShares(apiBase)),
    vscode.commands.registerCommand('axon.refreshIngest', () => refreshIngest(apiBase)),
    vscode.commands.registerCommand('axon.listStaleDocs', () => listStaleDocs(apiBase)),
    vscode.commands.registerCommand('axon.clearKnowledgeBase', () => clearKnowledgeBase(apiBase)),
    vscode.commands.registerCommand('axon.showGraphStatus', () => showGraphStatus(apiBase)),
    vscode.commands.registerCommand('axon.showGraphForQuery', async () => {
      const query = await vscode.window.showInputBox({ prompt: 'Axon: Enter query to visualise' });
      if (query) { await showGraphForQuery(context, query); }
    }),
    vscode.commands.registerCommand('axon.showGraphForSelection', () => showGraphForSelection(context)),
  );

  // Register Language Model Tools (for Copilot Agent toolset)
  try {
    if ('lm' in vscode && (vscode as any).lm.registerTool) {
      state.outputChannel.appendLine('Registering Axon Language Model Tools...');
      context.subscriptions.push(
        (vscode as any).lm.registerTool('axon_searchKnowledge', new AxonSearchTool()),
        (vscode as any).lm.registerTool('axon_queryKnowledge', new AxonQueryTool()),
        (vscode as any).lm.registerTool('axon_ingestText', new AxonIngestTextTool()),
        (vscode as any).lm.registerTool('axon_ingestUrl', new AxonIngestUrlTool()),
        (vscode as any).lm.registerTool('axon_ingestPath', new AxonIngestPathTool()),
        (vscode as any).lm.registerTool('axon_getIngestStatus', new AxonGetIngestStatusTool()),
        (vscode as any).lm.registerTool('axon_listProjects', new AxonListProjectsTool()),
        (vscode as any).lm.registerTool('axon_switchProject', new AxonSwitchProjectTool()),
        (vscode as any).lm.registerTool('axon_createProject', new AxonCreateProjectTool()),
        (vscode as any).lm.registerTool('axon_deleteProject', new AxonDeleteProjectTool()),
        (vscode as any).lm.registerTool('axon_deleteDocuments', new AxonDeleteDocumentsTool()),
        (vscode as any).lm.registerTool('axon_getCollection', new AxonGetCollectionTool()),
        (vscode as any).lm.registerTool('axon_clearCollection', new AxonClearCollectionTool()),
        (vscode as any).lm.registerTool('axon_updateSettings', new AxonUpdateSettingsTool()),
        (vscode as any).lm.registerTool('axon_shareProject', new AxonShareProjectTool()),
        (vscode as any).lm.registerTool('axon_redeemShare', new AxonRedeemShareTool()),
        (vscode as any).lm.registerTool('axon_revokeShare', new AxonRevokeShareTool()),
        (vscode as any).lm.registerTool('axon_listShares', new AxonListSharesTool()),
        (vscode as any).lm.registerTool('axon_initStore', new AxonInitStoreTool()),
        (vscode as any).lm.registerTool('axon_ingestImage', new AxonIngestImageTool()),
        (vscode as any).lm.registerTool('axon_refreshIngest', new AxonRefreshIngestTool()),
        (vscode as any).lm.registerTool('axon_listStaleDocs', new AxonListStaleDocsTool()),
        (vscode as any).lm.registerTool('axon_clearKnowledgeBase', new AxonClearKnowledgeBaseTool()),
        (vscode as any).lm.registerTool('axon_showGraphStatus', new AxonShowGraphStatusTool()),
        (vscode as any).lm.registerTool('axon_showGraph', new AxonShowGraphTool(context)),
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
