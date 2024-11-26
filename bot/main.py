from pypdf import PdfReader
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv('OPENAI_API_KEY')
st.set_page_config(page_title="Chatbot", layout="wide")
api_key = st.text_input("enter your openai API key",type="password",key="api_key_input")

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
  return text
        

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question,api_key):
    model = ChatOpenAI(model="gpt-3.5-turbo-1106", api_key=api_key)
    prompts = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>
    Question: {input}""")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompts)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    response =  chain({"input_documents": docs, "input": user_question}, return_only_outputs=True)
    st.write("Reply: ",response["output_text"])

def main():
    st.header("Bot App")
    user_question = st.text_input("ask a question from PDF files", key="user_question")
    if user_question and api_key:
        user_input(user_question, api_key)
    with st.sidebar:
        st.title("menu:")
        pdf_docs = st.file_uploader("upload your docs", accept_multiple_files=True, key="pdf_uploader")
        if st.button("submit & process",key="process_button") and api_key:
            with st.spinner("processing..."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("done")

if __name__ == "__main__":
    main()