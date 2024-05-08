# Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import tempfile
from bs4 import BeautifulSoup
# from PIL import Image

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to fetch PDF links from a given URL
def fetch_pdf_links(url):
    # Fetch webpage content
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch webpage content: {response.status_code}")
        return []

    # Parse HTML content to find links
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a')
    
    pdf_links = []
    # Extract PDF links from anchor tags
    for link in links:
        href = link.get('href', '')
        if '.pdf' in href:
            if not href.startswith('http'):
                href = url + href
            pdf_links.append(href)
    
    return pdf_links

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create or update vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def update_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = load_vector_store()
    vector_store.add_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Function to get conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    # Load chat model for question answering
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate response
def user_input(user_question):
    new_db = load_vector_store()
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Function to download PDF from URL
def download_pdf(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to download PDF from URL: {url}")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(response.content)
        return tf.name

# Function to extract text from PDF
def get_pdf_text(pdf_path):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Main function to define Streamlit app
def main():
    # Configure Streamlit page
    st.set_page_config("Research Paper Analyser", page_icon="favicon.png")
    col1, mid, col2 = st.columns([1,2,20])
    with col1:
        st.image('header.jpg', width=90)
    with col2:
        st.header("RESEARCH PAPER ANALYSER")
    user_question = st.text_input("Ask anything about the pdf files.")

    # Process user input and generate response
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Enter a URL or Upload PDF(s)")
        option = st.selectbox("Choose an option", ["URL", "Upload"])

        # Handle URL option
        if option == "URL":
            url = st.text_input("Enter the URL to fetch PDFs:")
            
            if st.button("Fetch and Process PDFs"):
                pdf_links = fetch_pdf_links(url)
                if not pdf_links:
                    st.warning("No URL provided.")
                    return
                
                st.info(f"Found {len(pdf_links)} URLs.")
                
                for idx, pdf_link in enumerate(pdf_links, 1):
                    st.write(f"Processing PDF {idx}/{len(pdf_links)}...")
                    
                    pdf_docs = download_pdf(pdf_link)
                    if pdf_docs:
                        with st.spinner("Processing..."):
                            raw_text = get_pdf_text(pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Done")
                    else:
                        st.warning("Failed to process the PDF")
        
        # Handle file upload option
        elif option == "Upload":
            st.write("Upload one or more PDF files:")
            uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with st.spinner("Processing..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                            tf.write(uploaded_file.getbuffer())
                            pdf_docs = tf.name
                            
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        update_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
