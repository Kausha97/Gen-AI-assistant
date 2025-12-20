import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# ----------------- GOOGLE API -----------------
# Set your Google GenAI API key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyDSO-1IXDnJZW5SYETGPjaXVC3nKYtdzic"

# ----------------- PDF PROCESSING -----------------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split large text into chunks for embeddings"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    """Create FAISS vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

# ----------------- QA CHAIN -----------------
def get_qa_chain():
    """Load a question-answering chain using Google GenAI"""
    template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the provided context, say "Answer not available in the context."

Context:\n{context}\n
Question:\n{question}\n
Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(question):
    """Process user question and display response"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings)
    docs = db.similarity_search(question)
    chain = get_qa_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("Reply:", result["output_text"])

# ----------------- STREAMLIT APP -----------------
def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.title("📚 Multi-PDF Chat Agent 🤖")

    # User question input
    user_question = st.text_input("Ask a question from the uploaded PDFs .. ✍️")
    if user_question:
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

    # Sidebar for PDF uploads
    with st.sidebar:
        st.header("📁 PDF Section")
        pdf_docs = st.file_uploader("Upload PDF Files & click Submit", accept_multiple_files=True)
        if pdf_docs and st.button("Submit & Process"):
            try:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
                    st.success("PDFs processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

        st.write("---")
        st.write("AI App created by @ Gurpreet Kaur")
        st.markdown(
            """
            <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 10px; text-align: center; color: white;">
            © Gurpreet Kaur | Made with ❤️
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
