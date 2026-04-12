/**
 * Axon Chat Controller
 * Handles conversation rendering, streaming, and UI interaction.
 */

class AxonChat {
    constructor(api, messagesContainer, inputElement, sendBtn) {
        this.api = api;
        this.container = messagesContainer;
        this.input = inputElement;
        this.sendBtn = sendBtn;
        this.messages = [];
        this.isStreaming = false;

        // Diagnostic panel elements
        this.diagnosticPanel = document.getElementById('diagnostic-panel');
        this.diagnosticSteps = document.getElementById('diagnostic-steps');

        this.setupListeners();
    }

    setupListeners() {
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });

        // Quick actions
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.input.value = btn.dataset.query;
                this.handleSendMessage();
            });
        });
    }

    async handleSendMessage() {
        const text = this.input.value.trim();
        if (!text || this.isStreaming) return;

        this.input.value = '';
        this.input.style.height = 'auto';

        // Hide welcome message if it's there
        const welcome = this.container.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        this.addMessage('user', text);

        // Prep assistant response
        const assistantMsgDiv = this.addMessage('assistant', '');
        const contentDiv = assistantMsgDiv.querySelector('.message-content');
        this.isStreaming = true;

        // Clear diagnostics
        this.clearDiagnostics();
        this.diagnosticPanel.classList.remove('hidden');

        let fullContent = '';
        let sources = [];

        try {
            await this.api.streamQuery(
                text,
                {},
                (tokenData) => {
                    if (tokenData.type === 'token') {
                        fullContent += tokenData.content;
                        contentDiv.innerHTML = marked.parse(fullContent);
                    } else if (tokenData.type === 'sources') {
                        sources = tokenData.content;
                        this.renderSources(assistantMsgDiv, sources);
                    }
                    this.container.scrollTop = this.container.scrollHeight;
                },
                (diagnostic) => {
                    this.addDiagnosticStep(diagnostic);
                }
            );
        } catch (error) {
            console.error('Chat error:', error);
            contentDiv.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        } finally {
            this.isStreaming = true; // Wait for any trailing data? No, it's done.
            this.isStreaming = false;
        }
    }

    renderSources(messageDiv, sources) {
        if (!sources || sources.length === 0) return;

        let sourcesDiv = messageDiv.querySelector('.sources-expander');
        if (!sourcesDiv) {
            sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources-expander';
            messageDiv.querySelector('.message-content').appendChild(sourcesDiv);
        }

        const sourcesHtml = sources.map((s, i) => `
            <div class="source-card">
                <div class="source-header">
                    <span class="source-index">[${i+1}]</span>
                    <span class="source-name">${s.metadata?.source || 'Document'}</span>
                    <span class="source-score">${(s.score * 100).toFixed(0)}% Match</span>
                </div>
                <div class="source-snippet">${s.page_content.substring(0, 150)}...</div>
            </div>
        `).join('');

        sourcesDiv.innerHTML = `
            <details>
                <summary><i class="fas fa-book"></i> Sources (${sources.length})</summary>
                <div class="sources-list">${sourcesHtml}</div>
            </details>
        `;
    }

    addMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'assistant' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = role === 'assistant' ? marked.parse(text) : text;

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);

        this.container.appendChild(msgDiv);
        this.container.scrollTop = this.container.scrollHeight;

        this.messages.push({ role, content: text });
        return msgDiv;
    }

    addDiagnosticStep(step) {
        const li = document.createElement('li');
        li.innerHTML = `<div class="spinner"></div> <span>${step}</span>`;
        this.diagnosticSteps.appendChild(li);

        // If there's a previous step, replace its spinner with a check
        const previous = li.previousElementSibling;
        if (previous) {
            const spinner = previous.querySelector('.spinner');
            if (spinner) {
                spinner.className = 'fas fa-check-circle';
                spinner.style.fontSize = '12px';
            }
        }
    }

    clearDiagnostics() {
        this.diagnosticSteps.innerHTML = '';
    }

    clearHistory() {
        this.container.innerHTML = '';
        this.messages = [];
        // Add welcome back
        // ...
    }
}

export default AxonChat;
