from pypdf import PdfReader
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import dspy

# ColBERTv2 retrieval setup

from dotenv import load_dotenv
load_dotenv()
import os

# Set up Streamlit and load API key
st.set_page_config(page_title="Chatbot", layout="wide")
api_key = st.text_input("enter your openai API key",type="password",key="api_key_input")


# Extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    return text_splitter.split_text(text)

# Create and save a vector store
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Process user input and generate response
def user_input(user_question,api_key):   
    lm = dspy.LM('openai/gpt-4o-mini',api_key=api_key)
    dspy.configure(lm=lm)
    qa = dspy.Predict('question: str -> response: str')
    response =  qa(question=user_question)
    st.write("Reply: ",response.response)

# Main app function
def main():
    st.header("Bot App")
    user_question = st.text_input("Ask a question from PDF files", key="user_question")
    if user_question and api_key:
        user_input(user_question,api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your documents", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")

if __name__ == "__main__":
    main()
