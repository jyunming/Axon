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

        this.init();
    }

    async init() {
        try {
            // Check API health
            await fetch('http://localhost:8000/health');
        } catch (e) {
            console.error('FastAPI server not found at port 8000');
            // Show alert or handle
        }

        // Initialize Controllers
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
        this.setupProjectSelector();
        this.loadInitialData();

        this.isInitialized = true;
    }

    setupNavigation() {
        const navItems = document.querySelectorAll('.sidebar-nav li');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const viewId = item.dataset.view;
                this.switchView(viewId, item);
            });
        });
    }

    switchView(viewId, navItem) {
        if (this.currentView === viewId) return;

        // Update Sidebar
        document.querySelectorAll('.sidebar-nav li').forEach(i => i.classList.remove('active'));
        navItem.classList.add('active');

        // Update Section Visibility
        document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
        document.getElementById(`${viewId}-view`).classList.remove('hidden');

        this.currentView = viewId;

        // Specialized initialization
        if (viewId === 'graph') {
            if (!this.graph.graph) {
                this.graph.init();
            } else {
                this.graph.refresh();
            }
        } else if (viewId === 'files') {
            this.loadFiles();
        } else if (viewId === 'governance') {
            this.loadAuditLogs();
        }
    }

    async setupProjectSelector() {
        const selector = document.getElementById('project-selector');
        try {
            const projects = await api.listProjects();
            selector.innerHTML = '';

            projects.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.name;
                opt.textContent = p.name;
                selector.appendChild(opt);
            });

            selector.addEventListener('change', (e) => {
                const projectName = e.target.value;
                api.setProject(projectName);
                // Refresh current view if needed
                if (this.currentView === 'graph') this.graph.refresh();
                if (this.currentView === 'files') this.loadFiles();
            });
        } catch (e) {
            console.warn('Failed to load project list');
        }
    }

    async loadInitialData() {
        try {
            const config = await api.getConfig();
            // Fill settings etc.
            if (config.llm_provider) document.getElementById('llm-provider').value = config.llm_provider;
            if (config.llm_model) document.getElementById('llm-model').value = config.llm_model;
        } catch (e) {
            console.warn('Failed to load config');
        }
    }

    async loadFiles() {
        const fileList = document.getElementById('file-list');
        const totalDocs = document.getElementById('total-docs');
        const totalChunks = document.getElementById('total-chunks');

        try {
            const status = await api.getCollection();
            fileList.innerHTML = '';

            if (!status.files || status.files.length === 0) {
                fileList.innerHTML = '<div class="empty-state"><i class="far fa-file-alt"></i><p>No documents found.</p></div>';
                return;
            }

            totalDocs.textContent = status.total_files || status.files.length;
            totalChunks.textContent = status.total_chunks || 0;

            status.files.forEach(f => {
                const item = document.createElement('div');
                item.className = 'file-item';
                item.innerHTML = `
                    <span class="file-name">${f.source || 'Document'}</span>
                    <span class="file-type">${f.source.split('.').pop().toUpperCase()}</span>
                    <span class="file-status">Ingested</span>
                    <span class="file-date">${f.chunks} chunks</span>
                `;
                fileList.appendChild(item);
            });
        } catch (e) {
            fileList.innerHTML = '<div class="empty-state">Error loading Knowledge Base</div>';
            console.error('KB Error:', e);
        }
    }

    async loadAuditLogs() {
        const logsContainer = document.getElementById('audit-logs');
        try {
            const logs = await api.getAuditLogs();
            logsContainer.innerHTML = '';

            if (logs.length === 0) {
                logsContainer.innerHTML = '<div class="empty-state"><i class="fas fa-shield-alt"></i><p>No audit logs.</p></div>';
                return;
            }

            logs.forEach(log => {
                const item = document.createElement('div');
                item.className = 'file-item'; // repurpose styling
                item.innerHTML = `
                    <span>${new Date(log.timestamp).toLocaleTimeString()}</span>
                    <span>${log.action}</span>
                    <span>${log.user}</span>
                    <span>${log.project || 'global'}</span>
                `;
                logsContainer.appendChild(item);
            });
        } catch (e) {
            logsContainer.innerHTML = '<div class="empty-state">Error loading logs</div>';
        }
    }
}

// Initialise App
window.addEventListener('scroll', () => {
    // Basic auto-expanding textarea
    const tx = document.getElementById('user-input');
    tx.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
});

document.addEventListener('DOMContentLoaded', () => {
    window.axon = new AxonApp();
});
