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

import { makeChatHandler, AxonSearchTool, AxonQueryTool } from './tools/query';

import {

  AxonIngestTextTool, AxonIngestUrlTool, AxonIngestPathTool,

  AxonGetIngestStatusTool, AxonIngestImageTool, AxonRefreshIngestTool,

  AxonGetStaleDocsTool, AxonClearKnowledgeTool, AxonIngestTextsTool,

  ingestCurrentFile, ingestWorkspaceFolder, ingestPickedFolder,

  refreshIngest, listStaleDocs, clearKnowledgeBase,

} from './tools/ingest';

import {

  AxonListProjectsTool, AxonSwitchProjectTool, AxonCreateProjectTool,

  AxonDeleteProjectTool, AxonDeleteDocumentsTool, AxonListKnowledgeTool,

  AxonUpdateSettingsTool, AxonGetCurrentSettingsTool,

  AxonListSessionsTool, AxonGetSessionTool,

  switchProject, createNewProject,

} from './tools/projects';

import {

  AxonShareProjectTool, AxonRedeemShareTool, AxonRevokeShareTool,

  AxonListSharesTool, AxonInitStoreTool, AxonGetStoreStatusTool,

  initStore, shareProject, redeemShare, revokeShare, listShares,

} from './tools/shares';

import {

  AxonGraphStatusTool, AxonShowGraphTool, AxonGraphFinalizeTool,

  AxonGraphDataTool, AxonGetActiveLeasesTool,

  showGraphStatus,

} from './tools/graph';

import { AxonConfigValidateTool, AxonConfigSetTool, runConfigSetupWizard } from './tools/config';

import { showGovernancePanel } from './governance/panel';

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

  const participant = vscode.chat.createChatParticipant('axon.chat', makeChatHandler(context));

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

    vscode.commands.registerCommand('axon.showGovernancePanel', () => showGovernancePanel(context)),

    vscode.commands.registerCommand('axon.configSetup', async () => {

      const cfg = vscode.workspace.getConfiguration('axon');

      const apiBase = cfg.get<string>('apiBase', 'http://127.0.0.1:8000');

      const apiKey = cfg.get<string>('apiKey', '');

      await runConfigSetupWizard(apiBase, apiKey);

    }),

  );

  // Register Language Model Tools (for Copilot Agent toolset)

  try {

    if ('lm' in vscode && (vscode as any).lm.registerTool) {

      state.outputChannel.appendLine('Registering Axon Language Model Tools...');

      context.subscriptions.push(

        (vscode as any).lm.registerTool('search_knowledge', new AxonSearchTool(context)),

        (vscode as any).lm.registerTool('query_knowledge', new AxonQueryTool(context)),

        (vscode as any).lm.registerTool('ingest_text', new AxonIngestTextTool()),

        (vscode as any).lm.registerTool('ingest_texts', new AxonIngestTextsTool()),

        (vscode as any).lm.registerTool('ingest_url', new AxonIngestUrlTool()),

        (vscode as any).lm.registerTool('ingest_path', new AxonIngestPathTool()),

        (vscode as any).lm.registerTool('get_job_status', new AxonGetIngestStatusTool()),

        (vscode as any).lm.registerTool('refresh_ingest', new AxonRefreshIngestTool()),

        (vscode as any).lm.registerTool('list_projects', new AxonListProjectsTool()),

        (vscode as any).lm.registerTool('switch_project', new AxonSwitchProjectTool()),

        (vscode as any).lm.registerTool('create_project', new AxonCreateProjectTool()),

        (vscode as any).lm.registerTool('delete_project', new AxonDeleteProjectTool()),

        (vscode as any).lm.registerTool('delete_documents', new AxonDeleteDocumentsTool()),

        (vscode as any).lm.registerTool('list_knowledge', new AxonListKnowledgeTool()),

        (vscode as any).lm.registerTool('clear_knowledge', new AxonClearKnowledgeTool()),

        (vscode as any).lm.registerTool('get_stale_docs', new AxonGetStaleDocsTool()),

        (vscode as any).lm.registerTool('update_settings', new AxonUpdateSettingsTool()),

        (vscode as any).lm.registerTool('get_current_settings', new AxonGetCurrentSettingsTool()),

        (vscode as any).lm.registerTool('list_sessions', new AxonListSessionsTool()),

        (vscode as any).lm.registerTool('get_session', new AxonGetSessionTool()),

        (vscode as any).lm.registerTool('share_project', new AxonShareProjectTool()),

        (vscode as any).lm.registerTool('redeem_share', new AxonRedeemShareTool()),

        (vscode as any).lm.registerTool('revoke_share', new AxonRevokeShareTool()),

        (vscode as any).lm.registerTool('list_shares', new AxonListSharesTool()),

        (vscode as any).lm.registerTool('init_store', new AxonInitStoreTool()),

        (vscode as any).lm.registerTool('get_store_status', new AxonGetStoreStatusTool()),

        (vscode as any).lm.registerTool('ingest_image', new AxonIngestImageTool()),

        (vscode as any).lm.registerTool('graph_status', new AxonGraphStatusTool()),

        (vscode as any).lm.registerTool('show_graph', new AxonShowGraphTool(context)),

        (vscode as any).lm.registerTool('graph_finalize', new AxonGraphFinalizeTool()),

        (vscode as any).lm.registerTool('graph_data', new AxonGraphDataTool()),

        (vscode as any).lm.registerTool('get_active_leases', new AxonGetActiveLeasesTool()),

        (vscode as any).lm.registerTool('axon_config_validate', new AxonConfigValidateTool()),

        (vscode as any).lm.registerTool('axon_config_set', new AxonConfigSetTool()),

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

