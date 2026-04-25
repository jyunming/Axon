/**
 * Axon Governance Panel — VS Code WebviewPanel for operator console.
 *
 * Polls GET /governance/overview every 30s and renders:
 *   - Project / maintenance state
 *   - Graph entity + community counts
 *   - Stale doc count
 *   - Active ingest jobs
 *   - Active Copilot sessions
 *
 * Operator buttons:
 *   - Rebuild Graph   → POST /governance/graph/rebuild
 *   - Set Maintenance → POST /governance/project/maintenance
 *   - Expire Session  → POST /governance/copilot/session/{id}/expire
 *
 * Pattern mirrors graph/panel.ts — singleton WebviewPanel with auto-refresh.
 */
import * as vscode from 'vscode';
import { httpGet, httpPost } from '../client/http';

const POLL_INTERVAL_MS = 30_000;
const PANEL_TITLE = 'Axon Governance';
const VIEW_TYPE = 'axon.governancePanel';

export interface GovernanceOverview {
  project: string;
  maintenance: { maintenance_state: string; active_leases: number };
  graph: { entity_count: number; relation_count: number; community_count: number };
  stale_doc_count: number;
  active_ingest_jobs: number;
  copilot_sessions_active: number;
  project_count: number;
}

export class AxonGovernancePanel {
  static currentPanel: AxonGovernancePanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _apiBase: string;
  private readonly _apiKey: string;
  private _disposables: vscode.Disposable[] = [];
  private _pollTimer: ReturnType<typeof setInterval> | undefined;
  private _disposed = false;
  private _lastOverview: GovernanceOverview | null = null;
  static createOrReveal(
    context: vscode.ExtensionContext,
    apiBase: string,
    apiKey: string,
  ): AxonGovernancePanel {
    if (AxonGovernancePanel.currentPanel) {
      AxonGovernancePanel.currentPanel._panel.reveal(vscode.ViewColumn.Two);
      return AxonGovernancePanel.currentPanel;
    }
    const panel = vscode.window.createWebviewPanel(
      VIEW_TYPE,
      PANEL_TITLE,
      vscode.ViewColumn.Two,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(context.extensionUri, 'media'),
        ],
      },
    );
    AxonGovernancePanel.currentPanel = new AxonGovernancePanel(
      panel,
      apiBase,
      apiKey,
    );
    return AxonGovernancePanel.currentPanel;
  }
  private constructor(
    panel: vscode.WebviewPanel,
    apiBase: string,
    apiKey: string,
  ) {
    this._panel = panel;
    this._apiBase = apiBase;
    this._apiKey = apiKey;
    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
    this._panel.webview.onDidReceiveMessage(
      (msg) => this._handleMessage(msg),
      null,
      this._disposables,
    );
    // Render loading state immediately, then fetch
    this._renderLoading();
    this._refresh().catch(() => {});
    // Auto-poll
    this._pollTimer = setInterval(() => {
      if (!this._disposed) {
        this._refresh().catch(() => {});
      }
    }, POLL_INTERVAL_MS);
  }
  private async _refresh(): Promise<void> {
    try {
      const res = await httpGet(`${this._apiBase}/governance/overview`, this._apiKey);
      if (res.status === 200) {
        try {
          const parsed = JSON.parse(res.body) as GovernanceOverview;
          this._lastOverview = parsed;
          this._panel.webview.html = this._buildHtml(parsed);
        } catch {
          vscode.window.showErrorMessage(
            'Axon: Failed to parse governance overview from server response.',
          );
          if (this._lastOverview) {
            this._panel.webview.html = this._buildHtml(this._lastOverview);
          } else {
            this._panel.webview.html =
              '<!DOCTYPE html><html><body style="font-family:var(--vscode-font-family);padding:24px;color:var(--vscode-editor-foreground);"><p>Failed to load Axon Governance Console overview. Please try again.</p></body></html>';
          }
        }
      }
    } catch {
      // Keep showing last known state on network error
    }
  }
  private _renderLoading(): void {
    this._panel.webview.html = `<!DOCTYPE html><html><body style="font-family:var(--vscode-font-family);padding:24px;color:var(--vscode-editor-foreground);">
      <p>Loading Axon Governance Console…</p></body></html>`;
  }
  private _handleMessage(msg: { command: string; [k: string]: unknown }): void {
    switch (msg.command) {
      case 'rebuildGraph':
        this._rebuildGraph();
        break;
      case 'setMaintenance':
        this._setMaintenance();
        break;
      case 'expireSession':
        this._expireSession(msg.sessionId as string);
        break;
      case 'refresh':
        this._refresh().catch(() => {});
        break;
    }
  }
  private async _rebuildGraph(): Promise<void> {
    try {
      const res = await httpPost(
        `${this._apiBase}/governance/graph/rebuild`,
        {},
        this._apiKey,
      );
      if (res.status === 200) {
        vscode.window.showInformationMessage('Axon: Graph rebuild started.');
        await this._refresh();
      } else {
        vscode.window.showErrorMessage(`Axon: Graph rebuild failed (${res.status}).`);
      }
    } catch (err) {
      vscode.window.showErrorMessage(`Axon: Graph rebuild error: ${err}`);
    }
  }
  private async _setMaintenance(): Promise<void> {
    const overview = this._lastOverview;
    const project = overview?.project ?? '';
    const state = await vscode.window.showQuickPick(
      ['normal', 'readonly', 'draining', 'offline'],
      { placeHolder: `Set maintenance state for '${project}'` },
    );
    if (!state) { return; }
    try {
      const res = await httpPost(
        `${this._apiBase}/governance/project/maintenance?name=${encodeURIComponent(project)}&state=${encodeURIComponent(state)}`,
        {},
        this._apiKey,
      );
      if (res.status === 200) {
        vscode.window.showInformationMessage(`Axon: Project '${project}' set to ${state}.`);
        await this._refresh();
      } else {
        vscode.window.showErrorMessage(`Axon: Maintenance update failed (${res.status}).`);
      }
    } catch (err) {
      vscode.window.showErrorMessage(`Axon: Maintenance error: ${err}`);
    }
  }
  private async _expireSession(sessionId: string): Promise<void> {
    if (!sessionId) { return; }
    try {
      const res = await httpPost(
        `${this._apiBase}/governance/copilot/session/${encodeURIComponent(sessionId)}/expire`,
        {},
        this._apiKey,
      );
      if (res.status === 200) {
        vscode.window.showInformationMessage(`Axon: Session '${sessionId}' expired.`);
        await this._refresh();
      } else {
        vscode.window.showErrorMessage(`Axon: Expire session failed (${res.status}).`);
      }
    } catch (err) {
      vscode.window.showErrorMessage(`Axon: Expire session error: ${err}`);
    }
  }
  private _buildHtml(ov: GovernanceOverview): string {
    const maintState = ov.maintenance?.maintenance_state ?? 'unknown';
    const leases = ov.maintenance?.active_leases ?? 0;
    const entities = ov.graph?.entity_count ?? 0;
    const communities = ov.graph?.community_count ?? 0;
    const stale = ov.stale_doc_count ?? 0;
    const jobs = ov.active_ingest_jobs ?? 0;
    const copilot = ov.copilot_sessions_active ?? 0;
    const projects = ov.project_count ?? 0;
    const maintColor = maintState === 'normal' ? '#4ec9b0'
      : maintState === 'draining' ? '#dcdcaa'
      : '#f44747';
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
<style>
  :root {
    --bg: var(--vscode-editor-background, #1e1e1e);
    --fg: var(--vscode-editor-foreground, #d4d4d4);
    --muted: var(--vscode-descriptionForeground, #888);
    --border: var(--vscode-panel-border, #333);
    --header-bg: var(--vscode-editorGroupHeader-tabsBackground, #252526);
    --btn-bg: var(--vscode-button-background, #0e639c);
    --btn-fg: var(--vscode-button-foreground, #fff);
    --btn-hover: var(--vscode-button-hoverBackground, #1177bb);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font-family: var(--vscode-font-family, sans-serif); font-size: 13px; padding: 16px; }
  h1 { font-size: 1em; font-weight: 600; margin-bottom: 16px; color: var(--fg); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card { background: var(--header-bg); border: 1px solid var(--border); border-radius: 4px; padding: 12px; }
  .card-label { font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 4px; }
  .card-value { font-size: 1.3em; font-weight: 600; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  button { background: var(--btn-bg); color: var(--btn-fg); border: none; border-radius: 3px; padding: 6px 14px; cursor: pointer; font-size: 0.85em; }
  button:hover { background: var(--btn-hover); }
  .footer { color: var(--muted); font-size: 0.75em; margin-top: 16px; }
</style>
</head>
<body>
<h1>Axon Governance Console — ${this._escHtml(ov.project)}</h1>

<div class="grid">
  <div class="card">
    <div class="card-label">Maintenance State</div>
    <div class="card-value"><span class="badge" style="background:${maintColor}20;color:${maintColor}">${this._escHtml(maintState)}</span></div>
  </div>
  <div class="card">
    <div class="card-label">Active Write Leases</div>
    <div class="card-value">${leases}</div>
  </div>
  <div class="card">
    <div class="card-label">Graph Entities</div>
    <div class="card-value">${entities}</div>
  </div>
  <div class="card">
    <div class="card-label">Communities</div>
    <div class="card-value">${communities}</div>
  </div>
  <div class="card">
    <div class="card-label">Stale Docs (30d)</div>
    <div class="card-value">${stale}</div>
  </div>
  <div class="card">
    <div class="card-label">Active Ingest Jobs</div>
    <div class="card-value">${jobs}</div>
  </div>
  <div class="card">
    <div class="card-label">Copilot Sessions</div>
    <div class="card-value">${copilot}</div>
  </div>
  <div class="card">
    <div class="card-label">Projects</div>
    <div class="card-value">${projects}</div>
  </div>
</div>

<div class="actions">
  <button onclick="vscode.postMessage({command:'rebuildGraph'})">Rebuild Graph</button>
  <button onclick="vscode.postMessage({command:'setMaintenance'})">Set Maintenance</button>
  <button onclick="vscode.postMessage({command:'refresh'})">Refresh Now</button>
</div>

<div class="footer">Auto-refreshes every 30s. Last update: ${new Date().toLocaleTimeString()}</div>

<script>
  const vscode = acquireVsCodeApi();
</script>
</body>
</html>`;
  }
  private _escHtml(s: string): string {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  dispose(): void {
    if (this._disposed) { return; }
    this._disposed = true;
    if (this._pollTimer) { clearInterval(this._pollTimer); }
    AxonGovernancePanel.currentPanel = undefined;
    this._panel.dispose();
    this._disposables.forEach(d => d.dispose());
    this._disposables = [];
  }
}

/** Show (or reveal) the Governance Panel. */
export async function showGovernancePanel(
  context: vscode.ExtensionContext,
): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const apiKey = config.get<string>('apiKey', '');
  AxonGovernancePanel.createOrReveal(context, apiBase, apiKey);
}
