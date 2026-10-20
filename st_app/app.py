import os
from pathlib import Path
import streamlit as st
from pages.home import show_home
from pages.pdf import show_pdf_page
from pages.chat import show_chat_page

APP_ICON = Path(__file__).parent / "assets" / "app_icon.png"

st.set_page_config(
    page_title="Hybrid Rag Assistant",
    page_icon=str(APP_ICON),
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

def landing_page():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .stApp {
            background: #f5f7fc;
        }

        .block-container {
            max-width: 1600px;
            padding: 24px 40px 60px;
        }

        /* Top navigation */
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 110px;
            width: 100%;
        }

        .brand {
            color: #0f172a;
            font-size: 16px;
            font-weight: 700;
            text-decoration: none;
        }

        .api-link {
            color: #5b6478;
            font-size: 14px;
            text-decoration: none;
        }

        .api-link:hover {
            color: #4f46e5;
        }

        /* Hero section */
        .hero {
            text-align: center;
            margin: 0 auto 74px;
        }

        .hero h1 {
            max-width: 900px;
            margin: 0 auto;
            color: #0f172a;
            font-size: clamp(42px, 5vw, 62px);
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -2px;
        }

        .hero p {
            max-width: 650px;
            margin: 24px auto 0;
            color: #5b6478;
            font-size: 17px;
            line-height: 1.8;
        }

        /* Landing page buttons */
        div[data-testid="stButton"] button,
        div[data-testid="stLinkButton"] a {
            min-height: 56px !important;
            height: 56px !important;
            border-radius: 11px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            transition: background-color 0.2s ease;
        }

        div[data-testid="stButton"] button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: #4f46e5 !important;
            color: #ffffff !important;
            border: 1px solid #4f46e5 !important;
        }

        div[data-testid="stButton"] button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background: #4338ca !important;
            border-color: #4338ca !important;
        }

        div[data-testid="stLinkButton"] a {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #e0e5ef !important;
            text-decoration: none !important;
        }

        /* Feature section */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 40px;
            border-top: 1px solid #dfe4ed;
            padding-top: 42px;
            margin-top: 0;
        }

        .feature h3 {
            color: #0f172a;
            font-size: 20px;
            font-weight: 600;
            margin: 0 0 8px;
        }

        .feature p {
            color: #5b6478;
            font-size: 16px;
            line-height: 1.6;
            margin: 0;
        }

        /* Responsive layout */
        @media (max-width: 700px) {
            .block-container {
                padding: 24px 20px 40px;
            }

            .top-nav {
                margin-bottom: 75px;
            }

            .hero h1 {
                font-size: 40px;
                letter-spacing: -1px;
            }

            .hero p {
                font-size: 15px;
            }

            .feature-grid {
                grid-template-columns: 1fr;
                gap: 28px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top navigation and hero
    st.markdown(
        f"""
        <div class="top-nav">
            <span class="brand">Hybrid RAG Assistant</span>
            <a class="api-link"
               href="{API_BASE_URL}/docs"
               target="_blank"
               rel="noopener noreferrer">
                API docs
            </a>
        </div>
        <section class="hero">
            <h1>Ask questions across all your PDFs</h1>
            <p>
                Upload your documents and get answers with the exact source
                pages, so you can check every claim.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, get_started, gap, read_docs, right = st.columns(
        [3, 1.2, 0.12, 1.4, 3],
        gap="small"
    )

    with get_started:
        if st.button(
            "Get started",
            type="primary",
            use_container_width=True,
            key="landing_set_started"
        ):
            st.session_state.current_page = "Home"
            st.rerun()

    with read_docs:
        st.link_button(
            "Read the docs",
            f"{API_BASE_URL}/docs",
            type="secondary",
            use_container_width=True
        )

    # Centered feature section matching the HTML landing page
    left_space, feature_container, right_space = st.columns(
        [1, 8, 1],
        gap="small",
    )

    with feature_container:
        st.markdown(
            '<hr style="border:0; border-top:1px solid #dfe4ed; '
            'margin:24px 0 42px;">',
            unsafe_allow_html=True,
        )

        feature_cols = st.columns(3, gap="large")

        with feature_cols[0]:
            st.markdown(
                """
                <div class="feature">
                    <h3>Upload</h3>
                    <p>Add one PDF or many.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with feature_cols[1]:
            st.markdown(
                """
                <div class="feature">
                    <h3>Search</h3>
                    <p>Finds the passages that match your question.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with feature_cols[2]:
            st.markdown(
                """
                <div class="feature">
                    <h3>Answer</h3>
                    <p>Writes a reply and links the sources.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

landing = st.Page(
    landing_page,
    title="Welcome",
    url_path="welcome",
    default=True
)

home = st.Page(
    lambda: show_home(API_BASE_URL),
    title="Home",
    url_path="home",
    default=False
)
documents = st.Page(
    lambda: show_pdf_page(API_BASE_URL),
    title="PDF Documents",
    url_path="pdf"
)
chat = st.Page(
    lambda: show_chat_page(API_BASE_URL),
    title="RAG CHAT",
    url_path="chat"
)


if "current_page" not in st.session_state:
    st.session_state.current_page = "Welcome"

page_map = {
    "Welcome": landing,
    "Home": home,
    "PDF Documents": documents,
    "RAG Chat": chat
}

selected_page = page_map.get(
    st.session_state.current_page,
    landing
)

navigation = st.navigation(
    [selected_page],
    position="hidden"
)

navigation.run()
