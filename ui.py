import hashlib
import io

import streamlit as st
from openai import BadRequestError, RateLimitError

from services.chunker import chunk_pages
from services.pdf_loader import load_pdf_pages
from services.rag_engine import rag_answer


st.set_page_config(
    page_title="Neon Archive | PDF Chat",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Preparing your document...")
def prepare_pdf(file_bytes):
    pages = load_pdf_pages(io.BytesIO(file_bytes))
    text = "\n\n".join(page_text for _, page_text in pages)
    return text, tuple(chunk_pages(pages))


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #080a0f;
            --panel: #11151d;
            --panel-soft: #151b25;
            --line: #252d3a;
            --cyan: #55e6ff;
            --violet: #9b7bff;
            --lime: #c8ff68;
            --text: #f4f7fb;
            --muted: #8d98aa;
        }

        .stApp {
            background:
                radial-gradient(circle at 78% 4%, rgba(85, 230, 255, .10), transparent 24rem),
                radial-gradient(circle at 18% 82%, rgba(155, 123, 255, .08), transparent 22rem),
                var(--ink);
            color: var(--text);
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { visibility: hidden; height: 0; }
        [data-testid="stDecoration"] { display: none; }
        [data-testid="stSidebar"] {
            background: rgba(13, 17, 24, .92);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebarContent"] { padding: 2rem 1.25rem; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding: 2.5rem 3rem 8rem;
        }
        [data-testid="stFileUploader"] section {
            border: 1px dashed rgba(85, 230, 255, .48);
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(85, 230, 255, .07), rgba(155, 123, 255, .07));
            padding: 1.65rem;
            transition: border-color .2s ease, background .2s ease, transform .2s ease;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: var(--cyan);
            background: linear-gradient(135deg, rgba(85, 230, 255, .12), rgba(155, 123, 255, .10));
            transform: translateY(-2px);
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted); }
        [data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text); }
        [data-testid="stFileUploader"] button {
            border: 1px solid rgba(85, 230, 255, .65);
            border-radius: 10px;
            background: rgba(85, 230, 255, .10);
            color: var(--cyan);
        }
        [data-testid="stChatInput"] {
            background: rgba(17, 21, 29, .88);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, .32), 0 0 28px rgba(85, 230, 255, .06);
        }
        [data-testid="stChatInput"]:focus-within { border-color: rgba(85, 230, 255, .72); }
        [data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; }
        [data-testid="stChatMessageContent"] {
            border-radius: 14px;
            padding: .15rem 1rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
            background: rgba(155, 123, 255, .13);
            border: 1px solid rgba(155, 123, 255, .24);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
            background: rgba(17, 21, 29, .72);
            border: 1px solid rgba(85, 230, 255, .12);
        }
        .brand { display: flex; align-items: center; gap: .7rem; margin-bottom: 2.6rem; }
        .brand-mark {
            width: 2.2rem; height: 2.2rem; display: grid; place-items: center;
            border-radius: 10px; color: var(--ink); background: var(--cyan);
            font: 700 1rem 'Space Grotesk', sans-serif;
            box-shadow: 0 0 22px rgba(85, 230, 255, .42);
        }
        .brand-name { color: var(--text); font: 600 1.02rem 'Space Grotesk', sans-serif; letter-spacing: .01em; }
        .eyebrow { color: var(--cyan); font: 600 .7rem 'IBM Plex Mono', monospace; letter-spacing: .12em; text-transform: uppercase; }
        .hero { padding: 1.2rem 0 2rem; animation: rise .55s ease both; }
        .hero h1 { margin: .45rem 0 .5rem; color: var(--text); font: 700 clamp(2.25rem, 5vw, 4.8rem)/.98 'Space Grotesk', sans-serif; letter-spacing: -.04em; }
        .hero p { max-width: 34rem; margin: 0; color: var(--muted); font-size: 1.03rem; line-height: 1.6; }
        .topline { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: .5rem; }
        .status { display: inline-flex; align-items: center; gap: .45rem; color: var(--muted); font: 500 .72rem 'IBM Plex Mono', monospace; }
        .status-dot { width: .45rem; height: .45rem; border-radius: 50%; background: var(--lime); box-shadow: 0 0 10px var(--lime); }
        .section-label { margin: 1.5rem 0 .65rem; color: var(--muted); font: 600 .7rem 'IBM Plex Mono', monospace; letter-spacing: .11em; text-transform: uppercase; }
        .upload-heading { margin: 0 0 .8rem; color: var(--text); font: 600 1.15rem 'Space Grotesk', sans-serif; }
        .upload-caption { color: var(--muted); font-size: .9rem; margin: 0 0 1rem; }
        .file-card { display: flex; align-items: center; gap: .85rem; padding: .8rem .95rem; margin: .8rem 0 1.3rem; border: 1px solid rgba(200, 255, 104, .22); border-radius: 12px; background: rgba(200, 255, 104, .06); }
        .file-icon { display: grid; place-items: center; width: 2.2rem; height: 2.2rem; border-radius: 8px; color: var(--lime); background: rgba(200, 255, 104, .12); font: 600 .72rem 'IBM Plex Mono', monospace; }
        .file-name { overflow: hidden; color: var(--text); font-size: .86rem; text-overflow: ellipsis; white-space: nowrap; }
        .file-state { color: var(--lime); font: 500 .67rem 'IBM Plex Mono', monospace; text-transform: uppercase; }
        .empty { padding: 4rem 1rem 2rem; text-align: center; animation: rise .65s .08s ease both; }
        .empty-icon { width: 5rem; height: 5rem; display: grid; place-items: center; margin: 0 auto 1.5rem; border: 1px solid rgba(85, 230, 255, .35); border-radius: 20px; color: var(--cyan); font: 600 1.3rem 'IBM Plex Mono', monospace; box-shadow: 0 0 35px rgba(85, 230, 255, .12); }
        .empty h2 { margin: 0 0 .5rem; color: var(--text); font: 600 1.45rem 'Space Grotesk', sans-serif; }
        .empty p { margin: 0; color: var(--muted); }
        .tip { padding: .75rem 0; border-bottom: 1px solid rgba(37, 45, 58, .7); color: var(--muted); font-size: .82rem; line-height: 1.45; }
        .tip strong { display: block; margin-bottom: .15rem; color: var(--text); font-weight: 600; }
        .sidebar-note { margin-top: 2rem; padding: .9rem; border: 1px solid var(--line); border-radius: 12px; color: var(--muted); font-size: .75rem; line-height: 1.5; }
        @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem 7rem; }
            .topline { align-items: flex-start; flex-direction: column; }
            .hero h1 { font-size: 3rem; }
            .empty { padding-top: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(document_name=None):
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">N</div><div class="brand-name">Neon Archive</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("New conversation", icon=":material/add:", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown('<div class="section-label">Current file</div>', unsafe_allow_html=True)
        if document_name:
            st.markdown(
                f'<div class="file-card"><div class="file-icon">PDF</div><div><div class="file-name">{document_name}</div><div class="file-state">Ready to explore</div></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No file selected yet")

        st.markdown('<div class="section-label">Quick tips</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip"><strong>Ask naturally</strong>Use complete questions for more useful answers.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip"><strong>Go specific</strong>Ask about names, dates, definitions, or key points.</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip"><strong>Stay curious</strong>Follow up to explore another part of the file.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-note">Your workspace is ready when you are. Drop in a PDF to start a focused conversation.</div>', unsafe_allow_html=True)


def run_ui():
    apply_theme()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_pdf = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )
    document_name = uploaded_pdf.name if uploaded_pdf else None
    render_sidebar(document_name)

    st.markdown(
        '<div class="topline"><div class="eyebrow">Private document workspace</div><div class="status"><span class="status-dot"></span>Assistant online</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><h1>Talk to your<br><span style="color:#55e6ff">documents.</span></h1><p>A calm, precise space for exploring the ideas inside every PDF.</p></div>',
        unsafe_allow_html=True,
    )

    if not uploaded_pdf:
        st.markdown('<div class="section-label">Start here</div><div class="upload-heading">Bring a document into focus</div><p class="upload-caption">Upload a PDF and begin asking questions in seconds.</p>', unsafe_allow_html=True)
        st.markdown('<div class="empty"><div class="empty-icon">PDF</div><h2>Your next conversation starts here</h2><p>Choose a document above to unlock your workspace.</p></div>', unsafe_allow_html=True)
        return

    file_bytes = uploaded_pdf.getvalue()
    document_key = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("document_key") != document_key:
        st.session_state.document_key = document_key
        st.session_state.messages = []

    text, chunks = prepare_pdf(file_bytes)
    if not text.strip():
        st.error("This PDF does not contain selectable text. Please choose a text-based PDF.")
        return

    index = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask anything about this document")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        try:
            answer = rag_answer(query, index, chunks)
        except RateLimitError:
            answer = "The assistant is busy right now. Please try again in a moment."
        except BadRequestError as error:
            if "content_filter" in str(error):
                answer = (
                    "This request was blocked by the safety filter. "
                    "Please try a different question about the document."
                )
            else:
                answer = "The request could not be completed. Please try again."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
