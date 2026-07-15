import streamlit as st
from services.pdf_loader import load_pdf
from services.chunker import chunk_text
from services.embeddings import embed_chunks
from services.vector_store import build_index
from services.rag_engine import rag_answer

def run_ui():
    st.title("PDF RAG Chatbot (Azure OpenAI)")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        st.success("PDF uploaded successfully.")

        text = load_pdf(uploaded_pdf)
        chunks = chunk_text(text)

        embeddings = embed_chunks(chunks)
        index = build_index(embeddings)

        query = st.chat_input("Ask a question based on the PDF")

        if query:
            with st.chat_message("user"):
                st.write(query)

            answer = rag_answer(query, index, chunks)

            with st.chat_message("assistant"):
                st.write(answer)
