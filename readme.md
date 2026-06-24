# RAG Chat

Chat with any PDF using AI. Ask questions and get answers based on the document's actual content.

## How it works
1. Load the PDF using PyPDFLoader — pages are extracted automatically
2. Split pages into smaller chunks using RecursiveCharacterTextSplitter
3. Convert chunks into vectors and store in FAISS vectorstore
4. When a question is asked, retrieve the most relevant chunks using similarity search
5. Send chunks + question to Groq LLM (Llama 3.1) as context
6. LLM answers based on the actual document content

## Tech Stack
- Python
- LangChain
- FAISS (vector database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq API (Llama 3.1)
- python-dotenv

## How to run locally
1. Clone the repo
2. Install dependencies: `pip install langchain langchain-community langchain-groq langchain-text-splitters faiss-cpu pypdf sentence-transformers python-dotenv`
3. Create a `.env` file with your Groq API key: `GROQ_API_KEY=your-api-key`
4. Add your PDF as `resume.pdf` in the root folder
5. Run: `python app.py`