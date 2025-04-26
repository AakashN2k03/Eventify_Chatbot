import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

STATIC_PDF_PATH = "mern_eventify_PDF.pdf" 

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the context."

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.3)
    # model = ChatGoogleGenerativeAI(model="chat-bison-001", temperature=0.3)


    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if the index file exists before loading
    if os.path.exists("faiss_index/index.faiss"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        response = chain.invoke(
    {"input_documents": docs, "question": user_question}
)

        st.write("Reply: ", response["output_text"])
    else:
        st.error("Index file not found. Please process the PDF first.")

def main():
    st.set_page_config("AI POWERED EVENT PLANNER")
    st.header("Chat with EVENTIFY😎 ")

    user_question = st.text_input("Enter your query")

    if user_question:
        user_input(user_question)

    # Process the static PDF file at startup
    with st.spinner("Processing be patient..."):
        raw_text = get_pdf_text(STATIC_PDF_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        

if __name__ == "__main__":
    main()
