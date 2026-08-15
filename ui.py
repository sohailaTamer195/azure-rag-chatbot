import streamlit as st
from openai import RateLimitError

from services.pdf_loader import load_pdf
from services.chunker import chunk_text
from services.embeddings import embed_chunks
from services.vector_store import build_index
from services.rag_engine import rag_answer

@st.cache_data(show_spinner="Creating document embeddings...")
def cached_embed_chunks(chunks):
    return embed_chunks(tuple(chunks))


def run_ui():
    st.title("PDF RAG Chatbot (Azure OpenAI)")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        st.success("PDF uploaded successfully.")

        text = load_pdf(uploaded_pdf)
        if not text.strip():
            st.error(
                "No selectable text was found in this PDF. "
                "Please upload a text-based PDF or run OCR on the scanned document first."
            )
            return

        chunks = chunk_text(text)

        if sum(len(chunk) for chunk in chunks) <= 20000:
            index = None
        else:
            try:
                embeddings = cached_embed_chunks(tuple(chunks))
                index = build_index(embeddings)
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
