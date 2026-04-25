/**
 * Axon WebGUI Main Entry Point
 */

import api from './api.js';
import AxonChat from './chat.js';
import AxonGraph from './graph_explorer.js';

class AxonApp {
    constructor() {
        this.currentView = 'chat';
        this.chat = null;
        this.graph = null;
        this.isInitialized = false;
        this.noticeHost = null;
        this.init();
    }
    async init() {
        this.updateServerStatus(false);
        try {
            const health = await api.getHealth();
            if (health?.project) {
                api.setProject(health.project);
            }
            this.updateServerStatus(true);
        } catch (error) {
            console.error('FastAPI server not reachable:', error);
            this.updateServerStatus(false);
            this.showNotice('FastAPI server not reachable from the WebGUI.', 'error', 6000);
        }
        this.chat = new AxonChat(
            api,
            document.getElementById('chat-messages'),
            document.getElementById('user-input'),
            document.getElementById('send-btn')
        );
        this.graph = new AxonGraph(
            api,
            document.getElementById('graph-container')
        );
        this.setupNavigation();
        this.setupActions();
        this.setupTextareaResize();
        await this.setupProjectSelector();
        await this.loadInitialData();
        this.isInitialized = true;
    }
    updateServerStatus(isOnline) {
        const dot = document.querySelector('.status-indicator .dot');
        const text = document.querySelector('.status-text');
        if (!dot || !text) return;
        dot.classList.toggle('online', isOnline);
        dot.classList.toggle('offline', !isOnline);
        text.textContent = isOnline ? 'Server: Online' : 'Server: Offline';
    }
    ensureNoticeHost() {
        if (this.noticeHost) return this.noticeHost;
        const host = document.createElement('div');
        host.className = 'notice-stack';
        document.body.appendChild(host);
        this.noticeHost = host;
        return host;
    }
    showNotice(message, kind = 'info', timeoutMs = 4000) {
        const host = this.ensureNoticeHost();
        const notice = document.createElement('div');
        notice.className = `notice ${kind}`;
        notice.textContent = message;
        host.appendChild(notice);
        window.setTimeout(() => {
            notice.remove();
            if (host.childElementCount === 0) {
                host.remove();
                this.noticeHost = null;
            }
        }, timeoutMs);
    }
    setupNavigation() {
        const navItems = document.querySelectorAll('.sidebar-nav li');
        navItems.forEach(item => {
            item.addEventListener('click', async () => {
                const viewId = item.dataset.view;
                await this.switchView(viewId, item);
            });
        });
    }
    async switchView(viewId, navItem) {
        if (this.currentView === viewId) return;
        document.querySelectorAll('.sidebar-nav li').forEach(i => i.classList.remove('active'));
        navItem.classList.add('active');
        document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
        document.getElementById(`${viewId}-view`).classList.remove('hidden');
        this.currentView = viewId;
        if (viewId === 'graph') {
            if (!this.graph.graph) {
                await this.graph.init();
            } else {
                await this.graph.refresh();
            }
        } else if (viewId === 'files') {
            await this.loadFiles();
        } else if (viewId === 'governance') {
            await this.loadAuditLogs();
        }
    }
    setupActions() {
        const clearChatBtn = document.getElementById('clear-chat');
        const saveSettingsBtn = document.getElementById('save-settings');
        const ingestUrlBtn = document.getElementById('ingest-url');
        const fileUploadInput = document.getElementById('file-upload');
        const refreshGraphBtn = document.getElementById('refresh-graph');
        clearChatBtn?.addEventListener('click', () => {
            this.chat.clearHistory();
            this.showNotice('Chat history cleared.', 'info');
        });
        saveSettingsBtn?.addEventListener('click', async () => {
            await this.saveSettings(saveSettingsBtn);
        });
        ingestUrlBtn?.addEventListener('click', async () => {
            await this.handleIngestURL();
        });
        fileUploadInput?.addEventListener('change', async (event) => {
            await this.handleFileUpload(event);
        });
        refreshGraphBtn?.addEventListener('click', async () => {
            try {
                if (!this.graph.graph) {
                    await this.graph.init();
                } else {
                    await this.graph.refresh();
                }
                this.showNotice('Graph refreshed.', 'success');
            } catch (error) {
                console.error('Graph refresh failed:', error);
                this.showNotice(`Graph refresh failed: ${error.message}`, 'error', 6000);
            }
        });
    }
    setupTextareaResize() {
        const input = document.getElementById('user-input');
        if (!input) return;
        const resize = () => {
            input.style.height = 'auto';
            input.style.height = `${input.scrollHeight}px`;
        };
        input.addEventListener('input', resize);
        resize();
    }
    async setupProjectSelector() {
        const selector = document.getElementById('project-selector');
        if (!selector) return;
        try {
            const projects = await api.listProjects();
            const uniqueProjects = Array.from(new Map(
                projects.map(project => [project.name, project])
            ).values());
            selector.innerHTML = '';
            if (uniqueProjects.length === 0) {
                const fallback = document.createElement('option');
                fallback.value = 'default';
                fallback.textContent = 'default';
                selector.appendChild(fallback);
            } else {
                uniqueProjects.forEach(project => {
                    const option = document.createElement('option');
                    option.value = project.name;
                    option.textContent = project.name;
                    selector.appendChild(option);
                });
            }
            selector.value = api.activeProject;
            if (selector.value !== api.activeProject) {
                selector.value = 'default';
            }
            selector.addEventListener('change', async (event) => {
                const nextProject = event.target.value;
                const previousProject = api.activeProject;
                event.target.disabled = true;
                try {
                    await api.switchProject(nextProject);
                    this.chat.clearHistory();
                    if (this.currentView === 'graph') {
                        if (!this.graph.graph) {
                            await this.graph.init();
                        } else {
                            await this.graph.refresh();
                        }
                    }
                    if (this.currentView === 'files') {
                        await this.loadFiles();
                    }
                    if (this.currentView === 'governance') {
                        await this.loadAuditLogs();
                    }
                    this.showNotice(`Active project switched to ${nextProject}.`, 'success');
                } catch (error) {
                    console.error('Project switch failed:', error);
                    api.setProject(previousProject);
                    event.target.value = previousProject;
                    this.showNotice(`Project switch failed: ${error.message}`, 'error', 6000);
                } finally {
                    event.target.disabled = false;
                }
            });
        } catch (error) {
            console.warn('Failed to load project list:', error);
            this.showNotice('Project list could not be loaded.', 'error', 5000);
        }
    }
    async loadInitialData() {
        try {
            const config = await api.getConfig();
            if (config.llm_provider) document.getElementById('llm-provider').value = config.llm_provider;
            if (config.llm_model) document.getElementById('llm-model').value = config.llm_model;
            document.getElementById('toggle-hybrid').checked = Boolean(config.hybrid_search);
            document.getElementById('toggle-graph').checked = Boolean(config.graph_rag);
            document.getElementById('toggle-hyde').checked = Boolean(config.hyde);
            document.getElementById('toggle-rerank').checked = Boolean(config.rerank);
            const selector = document.getElementById('project-selector');
            if (selector) {
                selector.value = api.activeProject;
            }
        } catch (error) {
            console.warn('Failed to load config:', error);
            this.showNotice('Settings could not be loaded.', 'error', 5000);
        }
    }
    collectSettings() {
        const llmModel = document.getElementById('llm-model').value.trim();
        const settings = {
            llm_provider: document.getElementById('llm-provider').value,
            hybrid_search: document.getElementById('toggle-hybrid').checked,
            graph_rag: document.getElementById('toggle-graph').checked,
            hyde: document.getElementById('toggle-hyde').checked,
            rerank: document.getElementById('toggle-rerank').checked
        };
        if (llmModel) {
            settings.llm_model = llmModel;
        }
        return settings;
    }
    async saveSettings(button) {
        const settings = this.collectSettings();
        const originalText = button?.textContent || '';
        if (button) {
            button.disabled = true;
            button.textContent = 'Saving...';
        }
        try {
            const result = await api.updateConfig(settings);
            const appliedCount = Array.isArray(result.applied) ? result.applied.length : 0;
            this.showNotice(
                appliedCount > 0 ? `Saved ${appliedCount} setting(s).` : 'Settings saved.',
                'success'
            );
        } catch (error) {
            console.error('Settings save failed:', error);
            this.showNotice(`Settings save failed: ${error.message}`, 'error', 7000);
        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = originalText;
            }
        }
    }
    async handleIngestURL() {
        const url = window.prompt('URL to ingest');
        if (!url) return;
        try {
            await api.ingestURL(url.trim());
            await this.loadFiles();
            this.showNotice(`Ingested ${url.trim()}.`, 'success');
        } catch (error) {
            console.error('URL ingest failed:', error);
            this.showNotice(`URL ingest failed: ${error.message}`, 'error', 7000);
        }
    }
    async handleFileUpload(event) {
        const files = Array.from(event.target.files || []);
        event.target.value = '';
        if (files.length === 0) return;
        let shouldRefreshFiles = false;
        try {
            const result = await api.uploadFiles(files);
            const ingestedFiles = result.ingested_files ?? 0;
            const unsupported = (result.files || []).filter(file => file.status === 'unsupported');
            const failed = (result.files || []).filter(file => file.status === 'error');
            if (ingestedFiles > 0) {
                shouldRefreshFiles = true;
                this.showNotice(
                    `Uploaded ${ingestedFiles} file(s), ${result.ingested_chunks ?? 0} chunk(s).`,
                    'success'
                );
            }
            if (unsupported.length > 0) {
                this.showNotice(
                    `Unsupported upload(s): ${unsupported.map(file => file.filename).join(', ')}`,
                    'info',
                    8000
                );
            }
            if (failed.length > 0) {
                this.showNotice(
                    `Upload failed: ${failed.map(file => `${file.filename} (${file.error})`).join(', ')}`,
                    'error',
                    9000
                );
            }
        } catch (error) {
            console.error('File upload failed:', error);
            this.showNotice(`File upload failed: ${error.message}`, 'error', 8000);
        }
        if (shouldRefreshFiles && this.currentView === 'files') {
            await this.loadFiles();
        }
    }
    createEmptyState(label, message) {
        const wrapper = document.createElement('div');
        wrapper.className = 'empty-state';
        const icon = document.createElement('span');
        icon.className = 'empty-icon';
        icon.textContent = label;
        const text = document.createElement('p');
        text.textContent = message;
        wrapper.appendChild(icon);
        wrapper.appendChild(text);
        return wrapper;
    }
    async loadFiles() {
        const fileList = document.getElementById('file-list');
        const totalDocs = document.getElementById('total-docs');
        const totalChunks = document.getElementById('total-chunks');
        try {
            const status = await api.getCollection();
            fileList.replaceChildren();
            totalDocs.textContent = status.total_files ?? (status.files || []).length;
            totalChunks.textContent = status.total_chunks ?? 0;
            if (!status.files || status.files.length === 0) {
                fileList.appendChild(this.createEmptyState('Doc', 'No documents found.'));
                return;
            }
            status.files.forEach(file => {
                const item = document.createElement('div');
                item.className = 'file-item';
                const sourceName = file.source || 'Document';
                const fileType = sourceName.includes('.')
                    ? sourceName.split('.').pop().toUpperCase()
                    : 'DOC';
                const name = document.createElement('span');
                name.className = 'file-name';
                name.textContent = sourceName;
                const type = document.createElement('span');
                type.className = 'file-type';
                type.textContent = fileType;
                const state = document.createElement('span');
                state.className = 'file-status';
                state.textContent = 'Ingested';
                const chunks = document.createElement('span');
                chunks.className = 'file-date';
                chunks.textContent = `${file.chunks} chunks`;
                item.append(name, type, state, chunks);
                fileList.appendChild(item);
            });
        } catch (error) {
            console.error('Knowledge base load failed:', error);
            totalDocs.textContent = '0';
            totalChunks.textContent = '0';
            fileList.replaceChildren(this.createEmptyState('Err', 'Error loading Knowledge Base.'));
        }
    }
    async loadAuditLogs() {
        const logsContainer = document.getElementById('audit-logs');
        try {
            const response = await api.getAuditLogs();
            const logs = Array.isArray(response) ? response : (response.events || []);
            logsContainer.replaceChildren();
            if (logs.length === 0) {
                logsContainer.appendChild(this.createEmptyState('Log', 'No audit logs.'));
                return;
            }
            logs.forEach(log => {
                const item = document.createElement('div');
                item.className = 'file-item';
                const timestamp = document.createElement('span');
                timestamp.textContent = log.timestamp
                    ? new Date(log.timestamp).toLocaleTimeString()
                    : 'Unknown';
                const action = document.createElement('span');
                action.textContent = log.action || 'unknown';
                const actor = document.createElement('span');
                actor.textContent = log.actor || log.user || 'system';
                const project = document.createElement('span');
                project.textContent = log.project || 'global';
                item.append(timestamp, action, actor, project);
                logsContainer.appendChild(item);
            });
        } catch (error) {
            console.error('Audit log load failed:', error);
            logsContainer.replaceChildren(this.createEmptyState('Err', 'Error loading logs.'));
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.axon = new AxonApp();
});
