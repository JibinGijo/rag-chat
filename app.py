from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages

def split_text(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    return chunks

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(chunks,embeddings)
    return vectorstore

def ask_question(vectorscore, question):
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
    retriever = vectorscore.as_retriever()
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    pages = load_pdf("yourpdf.pdf")
    chunks=split_text(pages)
    vectorstore=create_vectorstore(chunks)
    print("PDF loaded. Ask your questions!")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = ask_question(vectorstore, question)
        print(f"Bot: {answer}")