import html
import requests
import streamlit as st

def show_pdf_page(api_base_url: str):
    api_base_url = api_base_url.rstrip("/")
    st.markdown(
        """
        <style>
        
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none;
        }

        .block-container {
            max-width: 1100px;
            padding: 30px 40px 60px;
        }

        .pdf-page-title {
            color: #0f172a;
            font-size: 30px;
            font-weight: 800;
            margin-bottom: 45px;
        }

        .knowledge-title {
            color: #0f172a;
            font-size: 27px;
            font-weight: 750;
            text-align: center;
            margin-bottom: 8px;
        }

        .knowledge-description {
            color: #64748b;
            text-align: center;
            margin-bottom: 28px;
        }

        .document-card {
            min-height: 65px;
            padding: 20px 14px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow-wrap: anywhere;
        }

        .document-name {
            color: #334155;
            font-size: 14px;
            font-weight: 600;
            text-align: center;
            overflow-wrap: anywhere;
        }

        .upload-heading {
            color: #0f172a;
            font-size: 20px;
            font-weight: 700;
            margin-top: 30px;
            margin-bottom: 12px;
        }

        /* Button styling */
        div[data-testid="stButton"] > button {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;

            width: 100% !important;
            height: 52px !important;
            min-height: 52px !important;
            box-sizing: border-box !important;

            padding: 0 28px !important;
            margin: 0 !important;

            border-radius: 12px !important;
            border: 1px solid #4f46e5 !important;

            background-color: #4f46e5 !important;
            color: #ffffff !important;

            font-size: 15px !important;
            font-weight: 600 !important;

            transition: background-color 0.2s ease;
        }

        div[data-testid="stButton"] > button:hover {
            background-color: #4338ca !important;
            border-color: #4338ca !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown(
            '<div class="pdf-page-title">PDF Documents</div>',
            unsafe_allow_html=True
        )
    with top_right:
        if st.button("Home", use_container_width=True):
            st.session_state.current_page = "Home"
            st.rerun()

    st.markdown(
        """
        <div class="knowledge-title">Your Knowledge Base</div>
        <div class="knowledge-description">
            View your documents and add PDFs to your knowledge base.
        </div>
        """,
        unsafe_allow_html=True
    )

    documents = []
    try:
        response = requests.get(
            f"{api_base_url}/api/documents/",
            timeout=15
        )
        response.raise_for_status()
        documents = response.json().get("documents", [])
    except requests.RequestException as e:
         st.error(f"Could not load your knowledge base: {e}")

    if documents:
        columns_per_row = 4
        for start in range(0, len(documents), columns_per_row):
            document_columns = st.columns(columns_per_row)
            for column, document in zip(
                document_columns,
                documents[start:start + columns_per_row]
            ):
                with column:
                    file_name = html.escape(
                        str(
                            document.get(
                                "file_name",
                                "Unnamed File"
                            )
                        )
                    )
                    status = document.get("status", "Unknown")
                    st.markdown(
                        f"""
                        <div class="document-card">
                            <div class="document-icon">📄</div>
                            <div class="document-name">
                                {file_name}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption(f"Status: {status}")
    else:
        st.info(
            "Your knowledge base is empty. "
            "Upload a PDF below and click Analyze File to get started."
        )      

    st.markdown(
        '<div class="upload-heading">Upload a PDF</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose Pdf File",
        type=["pdf"],
        key="knowledge_base_pdf"
    )
    if uploaded_file is not None:
        st.caption(f"Uploaded File: {uploaded_file.name}")

    if st.button(
        "Analyze File",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    ):
        try:
            with st.spinner("Uploading and indexing your PDF..."):
                response = requests.post(
                    f"{api_base_url}/api/documents/upload",
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=300
                )
                response.raise_for_status()
            st.success("PDF processed successfully.")
            st.session_state.pop("knowledge_base_pdf", None)
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not analyze the PDF: {exc}")     