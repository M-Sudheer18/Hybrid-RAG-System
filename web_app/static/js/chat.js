document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');
    const chatNotice = document.getElementById('chatNotice');
    const knowledgeStatus = document.getElementById('knowledgeStatus');
    const knowledgeStatusText = document.getElementById('knowledgeStatusText');
    const clearChatButton = document.getElementById('clearChatButton');

    let hasDocuments = false;
    let isSending = false;
    let sessionId = getSessionId();

    function getSessionId() {
        try {
            let id = sessionStorage.getItem('hybrid_rag_session_id');

            if (!id) {
                id = createSessionId();
                sessionStorage.setItem('hybrid_rag_session_id', id);
            }

            return id;
        } catch {
            return createSessionId();
        }
    }

    function createSessionId() {
        return window.crypto?.randomUUID
            ? window.crypto.randomUUID()
            : `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    }

    function showNotice(message, uploadLink = false) {
        chatNotice.replaceChildren();

        const text = document.createElement('span');
        text.textContent = message;
        chatNotice.appendChild(text);

        if (uploadLink) {
            const link = document.createElement('a');
            link.href = '/pdf';
            link.textContent = ' Upload a PDF';
            chatNotice.appendChild(link);
        }

        chatNotice.hidden = false;
    }

    function hideNotice() {
        chatNotice.hidden = true;
        chatNotice.replaceChildren();
    }

    async function checkDocuments() {
        try {
            const response = await fetch('/api/documents/');

            if (!response.ok) {
                throw new Error('Unable to retrieve documents.');
            }

            const data = await response.json();
            const documents = Array.isArray(data)
                ? data
                : (data.documents || []);

            const indexedDocuments = documents.filter(
                doc => doc.status === 'completed'
            );

            hasDocuments = indexedDocuments.length > 0;

            if (hasDocuments) {
                knowledgeStatusText.textContent =
                    `${indexedDocuments.length} indexed document(s) available`;

                knowledgeStatus.classList.add('status-ready');
                messageInput.disabled = false;
                updateSendButton();
                hideNotice();
            } else {
                knowledgeStatusText.textContent = 'No indexed documents available';
                knowledgeStatus.classList.add('status-empty');
                messageInput.disabled = true;
                sendButton.disabled = true;

                showNotice(
                    'Upload and index a PDF before asking questions.',
                    true
                );
            }
        } catch (error) {
            console.error('Document check failed:', error);

            knowledgeStatusText.textContent = 'Unable to check knowledge base';
            knowledgeStatus.classList.add('status-error');
            messageInput.disabled = true;
            sendButton.disabled = true;

            showNotice(
                'Could not connect to the document service. Please refresh the page.'
            );
        }
    }

    function updateSendButton() {
        sendButton.disabled =
            isSending || !hasDocuments || !messageInput.value.trim();

        sendButton.textContent = isSending ? 'Sending...' : 'Send';
    }

    function addMessage(role, message, sources = []) {
        const welcome = document.getElementById('welcomeMessage');

        if (welcome) {
            welcome.remove();
        }

        const row = document.createElement('div');
        row.className = `chat-message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'YOU' : 'AI';

        const column = document.createElement('div');
        column.style.minWidth = '0';
        column.style.maxWidth = '100%';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        const messageBody = document.createElement('div');
        messageBody.className = 'message-text';

        if (
            role === 'assistant' &&
            window.marked &&
            window.DOMPurify
        ) {
            const normalizedMessage = String(message)
                .replace(/&#x20;/gi, ' ')
                .replace(/\\+(?=[*_-])/g, '');

            const renderedHtml = window.marked.parse(normalizedMessage, {
                breaks: true
            });

            messageBody.innerHTML =
                window.DOMPurify.sanitize(renderedHtml);
        } else {
            messageBody.textContent = message;
        }

        bubble.appendChild(messageBody);
        column.appendChild(bubble);

        if (role === 'assistant' && sources.length > 0) {
            const sourceSection = document.createElement('div');
            sourceSection.className = 'message-sources';

            const heading = document.createElement('h4');
            heading.textContent = 'Sources';
            sourceSection.appendChild(heading);

            sources.forEach(source => {
                const item = document.createElement('div');
                item.className = 'source-item';

                const metadata = source.metadata || {};
                const title = document.createElement('strong');

                title.textContent =
                    source.chunk_id || metadata.source || 'Document source';

                const details = document.createElement('span');
                const parts = [];

                if (metadata.source) {
                    parts.push(metadata.source);
                }

                if (metadata.page != null) {
                    parts.push(`Page ${metadata.page}`);
                }

                if (typeof source.score === 'number') {
                    parts.push(`Score: ${source.score.toFixed(3)}`);
                }

                details.textContent = parts.join(' · ') || 'Retrieved passage';

                item.append(title, details);
                sourceSection.appendChild(item);
            });

            column.appendChild(sourceSection);
        }

        row.append(avatar, column);
        chatMessages.appendChild(row);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage(event) {
        event.preventDefault();

        const query = messageInput.value.trim();

        if (!query || isSending || !hasDocuments) {
            return;
        }

        hideNotice();
        addMessage('user', query);

        messageInput.value = '';
        isSending = true;
        messageInput.disabled = true;
        typingIndicator.hidden = false;
        updateSendButton();

        try {
            const response = await fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    query: query
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail || 'Failed to generate an answer.'
                );
            }

            if (!data.answer) {
                throw new Error('The assistant returned an empty answer.');
            }

            if (data.session_id) {
                sessionId = data.session_id;

                try {
                    sessionStorage.setItem(
                        'hybrid_rag_session_id',
                        sessionId
                    );
                } catch (error) {
                    console.warn('Could not save session ID:', error);
                }
            }

            addMessage('assistant', data.answer, data.sources || []);
        } catch (error) {
            console.error('Chat request failed:', error);
            showNotice(error.message || 'An unexpected error occurred.');
        } finally {
            isSending = false;
            typingIndicator.hidden = true;
            messageInput.disabled = !hasDocuments;
            updateSendButton();
            messageInput.focus();
        }
    }

    function clearChat() {
        if (isSending) {
            return;
        }

        chatMessages.replaceChildren();

        const welcome = document.createElement('div');
        welcome.className = 'welcome-message';
        welcome.id = 'welcomeMessage';

        const avatar = document.createElement('div');
        avatar.className = 'assistant-avatar';
        avatar.textContent = 'AI';

        const content = document.createElement('div');
        content.className = 'message-content';

        const heading = document.createElement('h3');
        heading.textContent = 'Hello! How can I help you?';

        const paragraph = document.createElement('p');
        paragraph.textContent =
            'Ask a question about the PDFs in your knowledge base.';

        content.append(heading, paragraph);
        welcome.append(avatar, content);
        chatMessages.appendChild(welcome);

        sessionId = createSessionId();

        try {
            sessionStorage.setItem('hybrid_rag_session_id', sessionId);
        } catch (error) {
            console.warn('Could not save session ID:', error);
        }

        messageInput.value = '';
        hideNotice();
        updateSendButton();
        messageInput.focus();
    }

    chatForm.addEventListener('submit', sendMessage);

    messageInput.addEventListener('input', updateSendButton);

    messageInput.addEventListener('keydown', event => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();

            if (!sendButton.disabled) {
                chatForm.requestSubmit();
            }
        }
    });

    clearChatButton.addEventListener('click', clearChat);

    checkDocuments();
});