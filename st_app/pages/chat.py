import uuid
import html
import requests
import streamlit as st

def show_chat_page(api_base_url: str):
    
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f6fb;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 8rem; /* room so the pinned input never covers answers */
        }

        /* Chat header */
        .chat-heading {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        }

        .chat-eyebrow {
            color: #64748b;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.6px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .chat-heading h1 {
            color: #111827;
            font-size: 34px;
            font-weight: 700;
            margin: 0 0 8px 0;
        }

        .chat-heading p {
            color: #64748b;
            font-size: 16px;
            margin: 0;
        }

        /* Section headings */
        .chat-input-label {
            color: #111827;
            font-size: 19px;
            font-weight: 600;
            margin: 12px 0;
        }

        .chat-answer-title {
            color: #111827;
            font-size: 22px;
            font-weight: 700;
            margin-top: 28px;
            margin-bottom: 12px;
        }

        /* User and assistant messages */
        .chat-message {
            border-radius: 14px;
            padding: 17px 20px;
            margin: 10px 0 16px 0;
            line-height: 1.7;
            overflow-wrap: anywhere;
            white-space: normal;
        }

        .user-message {
            background: #eaf0ff;
            border: 1px solid #d6e2ff;
            color: #1e3a8a;
        }

        .assistant-message {
            background: #ffffff;
            border: 1px solid #e1e5ed;
            color: #1f2937;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
        }

        .message-label {
            font-size: 12px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            margin-bottom: 7px;
        }

        /* ---------- Claude-style chat input (st.chat_input) ---------- */

        /* Outer container */
        [data-testid="stChatInput"] {
            background: #ffffff !important;
            border: 1px solid #dfe4ed !important;
            border-radius: 22px !important;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06) !important;
            padding: 6px 8px !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #94a3b8 !important;
            box-shadow: 0 4px 22px rgba(15, 23, 42, 0.10) !important;
        }

        /* Remove inner backgrounds/borders so the outer box is the only outline */
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="base-input"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* The text area: starts tall, grows as you type (Shift+Enter = new line) */
        [data-testid="stChatInput"] textarea {
            font-size: 18px !important;
            line-height: 1.5 !important;
            min-height: 64px !important;   /* starting height */
            max-height: 280px !important;  /* stops growing here, then scrolls */
            padding: 16px 16px !important;
            background: transparent !important;
        }

        /* Send button */
        div[data-testid="stButton"] button {
            min-height: 50px !important;
            border-radius: 10px !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            background-color: #4f46e5 !important;
            color: #ffffff !important;
            border: 1px solid #4f46e5 !important;
            transition: background-color 0.2s ease;
        }

        div[data-testid="stButton"] button:hover {
            background-color: #4338ca !important;
            border-color: #4338ca !important;
            color: #ffffff !important;
        }

        /* Empty answer area */
        .empty-chat {
            background: #ffffff;
            border: 1px dashed #cbd5e1;
            border-radius: 16px;
            padding: 28px;
            text-align: center;
            color: #64748b;
            margin-top: 22px;
        }

        .empty-chat strong {
            color: #334155;
            font-size: 17px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initializing Canversion State 
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Header
    header_col, back_col, action_col = st.columns(
        [5, 1, 1.4],
        gap="small",
    )

    with header_col:
        st.markdown(
            """
            <div class="chat-heading">
                <div class="chat-eyebrow">Multi-PDF RAG</div>
                <h1>Chat</h1>
                <p>
                    Ask questions about your documents and get
                    answers grounded in your knowledge base.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with back_col:
        st.write("")
        if st.button(
            "← Back",
            use_container_width=True,
            key="chat_back_home",
        ):
            st.session_state.current_page = "Home"
            st.rerun()

    with action_col:
        st.write("")
        if st.button(
            "+ New Chat",
            use_container_width=True,
            key="chat_new_conversation",
        ):
            st.session_state.chat_session_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.rerun()
    # Question input
    st.markdown(
        '<div class="chat-input-label">Ask from your knowledge base</div>',
        unsafe_allow_html=True,
    )

    with st.form("chat_question_form", clear_on_submit=True):
        input_col, send_col = st.columns([12, 1])
        with input_col:
            query = st.text_input(
                "Your Question",
                placeholder="Enter Your Question Here..",
                label_visibility="collapsed"
            )
        with send_col:
            submitted = st.form_submit_button(
                "➤",
                type="primary",
                use_container_width=True
            )

    # Process question through FastAPI
    if submitted:
        query = query.strip()
        if not query:
            st.warning("Please Enter your Question First..")
        else:
            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": query
                }
            )
            endpoint = f"{api_base_url.rstrip('/')}/api/chat/"

            with st.spinner("Searching your Documents and Generating an answer.."):
                try:
                    response = requests.post(
                        endpoint,
                        json={
                            "session_id": st.session_state.chat_session_id,
                            "query": query
                        },
                        timeout=180
                    )
                    response.raise_for_status()
                    payload = response.json()

                    # Support common response-wrapper formats.
                    if isinstance(payload, dict):
                        if isinstance(payload.get("data"), dict):
                            payload = payload["data"]
                        elif isinstance(payload.get("result"), dict):
                            payload = payload["result"]

                    if not isinstance(payload, dict):
                        raise ValueError(
                            "The backend returned an unexpected response format."
                        )

                    answer = (
                        payload.get("answer")
                        or payload.get("response")
                        or payload.get("message")
                    )

                    if not answer:
                        raise ValueError(
                            "The backend response did not contain an answer. "
                            "Check the FastAPI chat route."
                        )
                    sources = payload.get("sources", [])

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": str(answer),
                            "sources": sources if isinstance(sources, list) else []
                        }
                    )
                except requests.HTTPError as e:
                    status_code = (
                        e.response.status_code
                        if e.response is not None
                        else "unknown"
                    )
                    detail = ""
                    if e.response is not None:
                        try:
                            detail = str(
                                e.response.json().get("detail", "")
                            )
                        except (ValueError, AttributeError):
                            detail = ""

                    error_message = (
                            f"Backend returned HTTP {status_code}."
                            + (f" Details: {detail}" if detail else "")
                        )
                    st.session_state.chat_messages.append(
                            {
                                "role": "assistant",
                                "content": error_message,
                                "sources": [],
                            }
                        )
                except requests.RequestException:
                        st.session_state.chat_messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "I couldn't connect to the FastAPI backend. "
                                    "Check that the backend is running and that "
                                    "the API base URL is correct."
                                ),
                                "sources": [],
                            }
                        )
                except (ValueError, TypeError) as exc:
                        st.session_state.chat_messages.append(
                            {
                                "role": "assistant",
                                "content": f"Could not process the response: {exc}",
                                "sources": [],
                            }
                        )
            st.rerun()

    # Answer and conversation history
    st.markdown(
        '<div class="chat-answer-title">Answer</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_messages:
        st.markdown(
            """
            <div class="empty-chat">
                <strong>Your answer will appear here.</strong>
                <br><br>
                Ask a question above to search your indexed PDF documents.
                Previous questions and answers will appear here too.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for index, message in enumerate(st.session_state.chat_messages):
            role = message.get("role", "")
            content = html.escape(str(message.get("content", "")))
            content = content.replace("\n", "<br>")

            if role == "user":
                st.markdown(
                    f"""
                    <div class="chat-message user-message">
                        <div class="message-label">Your Question</div>
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif role == "assistant":
                answer_text = str(message.get("content", ""))

                # Normalize Markdown that may have been escaped.
                answer_text = (
                    answer_text
                    .replace(r"\*\*", "**")
                    .replace(r"\*", "*")
                    .replace("\\\n", "\n")
                )

                # Render the answer as Markdown instead of embedding
                # it inside an HTML string.
                with st.container(border=True):
                    st.markdown(
                        '<div class="message-label">ANSWER</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(answer_text)

                # Keep your existing sources-display code directly here.
                sources_for_message = message.get("sources", [])

                if sources_for_message:
                    with st.expander(
                        f"Sources used ({len(sources_for_message)})",
                        expanded=False,
                    ):
                        for source_number, source_item in enumerate(
                            sources_for_message, start=1
                        ):
                            metadata = source_item.get("metadata", {}) or {}

                            filename = (
                                metadata.get("filename")
                                or metadata.get("file_name")
                                or metadata.get("source")
                                or "Document source"
                            )

                            page = (
                                metadata.get("page")
                                or metadata.get("page_number")
                            )

                            st.markdown(
                                f"**{source_number}. {filename}**"
                            )

                            if page is not None:
                                st.caption(f"Page: {page}")

                            if source_item.get("score") is not None:
                                st.caption(
                                    f"Relevance score: {source_item['score']}"
                                )

                            if source_number < len(sources_for_message):
                                st.divider()
