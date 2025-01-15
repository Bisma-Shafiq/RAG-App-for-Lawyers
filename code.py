import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure page
st.set_page_config(page_title="Legal Case Analysis App", layout="wide")
st.title("Legal Case Analysis Assistant")

# Initialize session state
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Sidebar with app information
with st.sidebar:
    st.markdown("### About")
    st.markdown("This app helps analyze legal cases using RAG (Retrieval Augmented Generation) technology.")
    st.markdown("Upload your legal document and ask questions to get detailed analysis.")

# Initialize Google API and models
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=GOOGLE_API_KEY,
    model="models/text-embedding-004",
    chunk_size=100,
    task_type="retrieval_document"
)

def format_legal_docs(docs):
    """Format legal documents with proper structure and citations"""
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        citation = doc.metadata.get('citation', 'No citation available')
        date = doc.metadata.get('date', 'No date available')
        court = doc.metadata.get('court', 'Court information not available')
        
        formatted_text = f"""
Case Document {i}
Citation: {citation}
Court: {court}
Date: {date}
---
{doc.page_content}
---"""
        formatted_docs.append(formatted_text)
    
    return "\n\n".join(formatted_docs)

def setup_rag_chain(vector_store):
    """Set up the RAG chain with the provided vector store"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert Legal Research Assistant specializing in case law analysis and legal drafting.
                     Given case law context and a question from the user, you should:
                     1. Analyze the legal principles and precedents from the provided cases
                     2. Identify relevant holdings and ratio decidendi
                     3. Provide well-reasoned legal analysis with proper citations
                     4. When drafting, follow proper legal writing conventions and structure
                     
                     Format your responses with:
                     - Clear headings for different sections
                     - Proper case citations
                     - Specific references to relevant passages from the context
                     - Clear distinction between holdings and obiter dicta
                     - Practical applications or implications when relevant"""),
        
        HumanMessagePromptTemplate.from_template("""Analyze the following legal materials and answer the question provided.
        
        Case Law Context: {context}
        
        Legal Query: {question}
        
        Analysis: """)
    ])
    
    output_parser = StrOutputParser()
    
    return (
        {
            "context": retriever | format_legal_docs,
            "question": RunnablePassthrough(),
            "metadata": lambda x: {
                "query_type": "legal_analysis",
                "timestamp": datetime.now().isoformat()
            }
        }
        | chat_template
        | llm_model
        | output_parser
    )

# Main app interface
st.markdown("### Upload Legal Document")
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    with st.spinner('Processing document...'):
        # Load and process the PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load_and_split()
        
        # Get pages
        pages = [doc.page_content for doc in documents]
        
        # Text splitting
        text_split = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        chunks = text_split.create_documents(pages)
        
        # Create vector store
        st.session_state.vector_store = FAISS.from_texts(pages, embedding=embedding_model)
        
        # Setup RAG chain
        st.session_state.rag_chain = setup_rag_chain(st.session_state.vector_store)
    
    st.success('Document processed successfully!')
    
    # Query section
    st.markdown("### Ask Questions")
    query = st.text_area("Enter your legal query:", height=100, 
                        placeholder="e.g., 'What are the main legal principles in this case?' or 'Create a legal draft for this case'")
    
    if st.button("Analyze"):
        if query and st.session_state.rag_chain:
            with st.spinner('Analyzing...'):
                response = st.session_state.rag_chain.invoke(query)
                st.markdown("### Analysis Results")
                st.markdown(response)
        else:
            st.warning("Please enter a query.")
else:
    st.info("Please upload a PDF document to begin analysis.")

# Cleanup
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")