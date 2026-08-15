import io

import streamlit as st
from openai import RateLimitError

from services.pdf_loader import load_pdf_pages
from services.chunker import chunk_pages
from services.embeddings import embed_chunks
from services.vector_store import build_index
from services.rag_engine import rag_answer

@st.cache_data(show_spinner="Reading PDF...")
def prepare_pdf(file_bytes):
    pages = load_pdf_pages(io.BytesIO(file_bytes))
    text = "\n\n".join(page_text for _, page_text in pages)
    return text, tuple(chunk_pages(pages))


@st.cache_data(show_spinner="Creating document embeddings...")
def cached_embed_chunks(chunks):
    return embed_chunks(tuple(chunks))


@st.cache_resource(show_spinner="Building search index...")
def cached_build_index(embeddings):
    return build_index(embeddings)


def run_ui():
    st.title("PDF RAG Chatbot (Azure OpenAI)")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        st.success("PDF uploaded successfully.")

        text, chunks = prepare_pdf(uploaded_pdf.getvalue())
        if not text.strip():
            st.error(
                "No selectable text was found in this PDF. "
                "Please upload a text-based PDF or run OCR on the scanned document first."
            )
            return

        if sum(len(chunk) for chunk in chunks) <= 20000:
            index = None
        else:
            try:
                embeddings = cached_embed_chunks(tuple(chunks))
                index = cached_build_index(embeddings)
            except RateLimitError:
                index = None
                st.warning(
                    "Semantic search is temporarily unavailable. "
                    "Using keyword search instead."
                )

        query = st.chat_input("Ask a question based on the PDF")

        if query:
            with st.chat_message("user"):
                st.write(query)

            try:
                answer = rag_answer(query, index, chunks)
            except RateLimitError:
                st.error(
                    "Azure OpenAI is rate-limiting this request. "
                    "Wait a moment and try again, or check the deployment quota."
                )
                return

            with st.chat_message("assistant"):
                st.write(answer)
