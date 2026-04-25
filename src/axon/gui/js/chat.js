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
        this.diagnosticPanel = document.getElementById('diagnostic-panel');
        this.diagnosticSteps = document.getElementById('diagnostic-steps');
        this.setupListeners();
    }
    setupListeners() {
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
        this.input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.handleSendMessage();
            }
        });
        this.container.addEventListener('click', (event) => {
            const actionButton = event.target.closest('.action-btn');
            if (!actionButton) return;
            this.input.value = actionButton.dataset.query || '';
            this.handleSendMessage();
        });
    }
    setStreamingState(isStreaming) {
        this.isStreaming = isStreaming;
        this.sendBtn.disabled = isStreaming;
    }
    createWelcomeMessage() {
        const welcome = document.createElement('div');
        welcome.className = 'welcome-message';
        const icon = document.createElement('div');
        icon.className = 'welcome-icon';
        icon.textContent = 'AI';
        const heading = document.createElement('h2');
        heading.textContent = 'How can I help you today?';
        const description = document.createElement('p');
        description.textContent = 'Axon is your local AI partner. Ask a question about your ingested documents or start by uploading a file.';
        const actions = document.createElement('div');
        actions.className = 'quick-actions';
        const quickActions = [
            ['Summarize project', 'Summarize my project'],
            ['Identify entities', 'What are the key entities?'],
            ['Compare documents', 'Compare recent documents']
        ];
        quickActions.forEach(([label, query]) => {
            const button = document.createElement('button');
            button.className = 'action-btn';
            button.dataset.query = query;
            button.textContent = label;
            actions.appendChild(button);
        });
        welcome.append(icon, heading, description, actions);
        return welcome;
    }
    addMessage(role, text) {
        const message = document.createElement('div');
        message.className = `message ${role}`;
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'assistant' ? 'AI' : 'You';
        const content = document.createElement('div');
        content.className = 'message-content';
        const textNode = document.createElement('div');
        textNode.className = 'message-text';
        textNode.textContent = text;
        content.appendChild(textNode);
        message.append(avatar, content);
        this.container.appendChild(message);
        this.container.scrollTop = this.container.scrollHeight;
        this.messages.push({ role, content: text });
        return message;
    }
    renderSources(messageDiv, sources) {
        if (!sources || sources.length === 0) return;
        const content = messageDiv.querySelector('.message-content');
        let sourcesDiv = messageDiv.querySelector('.sources-expander');
        if (!sourcesDiv) {
            sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources-expander';
            content.appendChild(sourcesDiv);
        }
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = `Sources (${sources.length})`;
        const list = document.createElement('div');
        list.className = 'sources-list';
        sources.forEach((source, index) => {
            const card = document.createElement('div');
            card.className = 'source-card';
            const header = document.createElement('div');
            header.className = 'source-header';
            const sourceIndex = document.createElement('span');
            sourceIndex.className = 'source-index';
            sourceIndex.textContent = `[${index + 1}]`;
            const sourceName = document.createElement('span');
            sourceName.className = 'source-name';
            sourceName.textContent = source.metadata?.source || source.source || 'Document';
            const score = document.createElement('span');
            score.className = 'source-score';
            score.textContent = Number.isFinite(source.score)
                ? `${(source.score * 100).toFixed(0)}% Match`
                : 'Match unavailable';
            const snippet = document.createElement('div');
            snippet.className = 'source-snippet';
            const excerpt = source.text || source.page_content || '';
            snippet.textContent = excerpt
                ? `${excerpt.substring(0, 150)}${excerpt.length > 150 ? '...' : ''}`
                : 'No excerpt available.';
            header.append(sourceIndex, sourceName, score);
            card.append(header, snippet);
            list.appendChild(card);
        });
        details.append(summary, list);
        sourcesDiv.replaceChildren(details);
    }
    addDiagnosticStep(step) {
        const item = document.createElement('li');
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        const text = document.createElement('span');
        text.textContent = step;
        item.append(spinner, text);
        this.diagnosticSteps.appendChild(item);
        const previous = item.previousElementSibling;
        if (previous) {
            const previousSpinner = previous.querySelector('.spinner');
            if (previousSpinner) {
                previousSpinner.className = 'step-complete';
                previousSpinner.textContent = 'OK';
            }
        }
    }
    clearDiagnostics() {
        this.diagnosticSteps.innerHTML = '';
    }
    clearHistory() {
        this.container.replaceChildren(this.createWelcomeMessage());
        this.messages = [];
        this.clearDiagnostics();
        this.diagnosticPanel.classList.add('hidden');
        this.setStreamingState(false);
        this.input.value = '';
        this.input.style.height = 'auto';
    }
    async handleSendMessage() {
        const text = this.input.value.trim();
        if (!text || this.isStreaming) return;
        this.input.value = '';
        this.input.style.height = 'auto';
        const welcome = this.container.querySelector('.welcome-message');
        if (welcome) welcome.remove();
        this.addMessage('user', text);
        const assistantMsgDiv = this.addMessage('assistant', '');
        const textDiv = assistantMsgDiv.querySelector('.message-text');
        this.setStreamingState(true);
        this.clearDiagnostics();
        this.diagnosticPanel.classList.remove('hidden');
        let fullContent = '';
        try {
            await this.api.streamQuery(
                text,
                {},
                (tokenData) => {
                    if (tokenData.type === 'token') {
                        fullContent += tokenData.content;
                        textDiv.textContent = fullContent;
                    } else if (tokenData.type === 'sources') {
                        this.renderSources(assistantMsgDiv, tokenData.content);
                    }
                    this.container.scrollTop = this.container.scrollHeight;
                },
                (diagnostic) => {
                    this.addDiagnosticStep(diagnostic);
                }
            );
        } catch (error) {
            console.error('Chat error:', error);
            textDiv.textContent = `Error: ${error.message}`;
        } finally {
            // Always release the send lock, even if streaming fails mid-response.
            this.setStreamingState(false);
            if (this.diagnosticSteps.childElementCount === 0) {
                this.diagnosticPanel.classList.add('hidden');
            }
        }
    }
}

export default AxonChat;
