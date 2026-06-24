
import tempfile

import streamlit as st

from app import *

st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a pdf " , type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path=temp_file.name

    pages = load_pdf(temp_path)
    chunks = split_text(pages)
    vectorstore=create_vectorstore(chunks)

    st.success("PDF processed! Ask your questions below.")

    question = st.text_input("Ask a question about your PDF")

    if question:
        answer = ask_question(vectorstore, question)
        st.write(answer)