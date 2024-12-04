from langchain_core.messages import HumanMessage, AIMessage
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Helper Functions
def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_web_text(url):
    """Fetch text content from a web page."""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return "\n".join(doc.page_content for doc in data)
    except Exception as e:
        st.error(f"Error loading web page: {e}")
        return ""

def get_text_chunks(text):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_retreiver_chain(vector_store):
    """Create a retriever chain."""
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up information relevant to the conversation.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag(history_retriever_chain):
    """Create a conversational RAG chain."""
    llm = ChatOpenAI()
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(history_retriever_chain, document_chain)

def get_response(user_input):
    """Generate a response from the conversational chain."""
    history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag(history_retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response["answer"]

# Main Application
st.header("Chat with PDF and Web")

with st.sidebar:
    st.title("Menu")
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True, key="pdf_uploader")
    web_url = st.text_input("Enter a URL")
    process_button = st.button("Submit & Process")

# Process Inputs
if process_button:
    if not pdf_docs and not web_url.strip():
        st.error("Please upload at least one PDF or provide a URL.")
    else:
        # Combine and process inputs
        raw_text = ""
        if pdf_docs:
            raw_text += get_pdf_text(pdf_docs)
        if web_url.strip():
            raw_text += get_web_text(web_url.strip())
        
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.success("Documents processed successfully!")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="I am a bot. How can I help you?")
    ]

# Chat Interface
user_question = st.chat_input("Type your message here:")
if user_question and user_question.strip():
    if "vector_store" in st.session_state and st.session_state.vector_store:
        response = get_response(user_question)
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
    else:
        response = "Please process documents or a URL first."
        st.session_state.chat_history.append(AIMessage(content=response))

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)
