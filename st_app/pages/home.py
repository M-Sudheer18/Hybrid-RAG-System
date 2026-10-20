import streamlit as st

def show_home(api_base_url: str):
    st.markdown(
        """
        <style>
        .home-header {
            text-align: center;
            margin: 55px auto 65px;
        }
        .home-header h1 {
            color: #0f172a;
            font-size: clamp(36px, 5vw, 52px);
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -1.5px;
            margin-bottom: 18px;
        }
        .home-header p {
            max-width: 620px;
            margin: 0 auto;
            color: #5b6478;
            font-size: 17px;
            line-height: 1.8;
        }
        .home-card {
            min-height: 210px;
            padding: 30px;
            border: 1px solid #e3e6ef;
            border-radius: 16px;
            background: #ffffff;
        }
        .home-card h2 {
            color: #0f172a;
            font-size: 23px;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .home-card p {
            color: #5b6478;
            font-size: 15px;
            line-height: 1.7;
            margin-bottom: 20px;
        }
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
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="home-header">
            <h1>Your Documents, Your Answers</h1>
            <p>
                Upload PDF documents, build your knowledge base,
                and ask questions to get answers grounded in your files.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    documents_column, chat_column = st.columns(2, gap="large")
    with documents_column:
        st.markdown(
            """
            <div class="home-card">
                <h2>PDF Documents</h2>
                <p>
                    Upload new PDFs, process their contents,
                    and index them for document-based question answering.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(
            "Open Documents",
            type="primary",
            use_container_width=True,
            key="open_documents"
        ):
            st.session_state.current_page = "PDF Documents"
            st.rerun()

    with chat_column:
        st.markdown(
            """
            <div class="home-card">
                <h2>RAG Chat</h2>
                <p>
                    Ask questions about your indexed documents
                    and receive answers with relevant source references.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(
            "Open Chat",
            use_container_width=True,
            key="open_chat"
        ):
            st.session_state.current_page = "RAG Chat"
            st.rerun()

    try:
        import requests
        response = requests.get(
            f"{api_base_url.rstrip('/')}/api/documents/",
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        documents = data.get("documents", [])
        indexed_count = sum(
            1 for document in documents
            if document.get("status") == "completed"
        )
        st.caption(
            f"Knowledge Base: {indexed_count} indexed document(s)"
        )
    except Exception:
        st.caption("Knowledge Base: Unable to retrieve document status")