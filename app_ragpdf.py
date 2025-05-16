# How to load multiple PDFs into vector Database
# How to split large documents into small chunks for better embeddings
# how to use retrieval augmented generations (RAG) with langchain chain that combines a vector store retriever + an llm (GROQ) + Embedding (Hugging Face) + a prompt template + conversational Q&A Chat + Unique Session ID Wise 
# process from upload till extraction
# Load PDF file -> convert their contents into vector embeddings -> implemented a chat history so that each conversation is remembered -> how the user session logic (session_ID) helps each user to maintain their own conversation flow
# PDF Q&A with RAG, Embeddings, Chat History, and Session Handling

import os
import time
import tempfile  # Temporarily store uploaded PDFs

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

##Langchain core Classes and utilities

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# LangChain memory for chat history

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt import

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain LLM & chaining utilities

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Text splitting & embeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Store

from langchain_chroma import Chroma

# PDF Loader
 
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables (GROQ API, Hugging Face Token)

load_dotenv()

# Streamlit Page setup

st.set_page_config(
    page_title="📄 RAG Q&A with PDF & Chat History",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG Q&A WITH PDF UPLOADS & CHAT HISTORY")

st.sidebar.header("👩🏻‍🔧 Configuration")
st.sidebar.write(
    "- Enter Your GROQ API Key \n"
    "- Upload PDFs On The Main Page \n"
    "- Ask Questions And See Chat History"
)

# API keys & embedding Setup

api_key = st.sidebar.text_input("Groq API Key", type = "password")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "") # for Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# only proceed if the user has entered GROQ Key

if not api_key:
    st.warning(" 🔑Please Enter Your GROQ API Key In The Sidebar To Continue.")
    st.stop()

# Instantiate The GROQ LLM

llm = ChatGroq(groq_api_key = api_key, model_name ="gemma2-9b-it")

# File Uploader: Allow Multiple PDFs Uploads

uploaded_files = st.file_uploader(
    "📁 Choose PDF Files(s)",
    type = "pdf",
    accept_multiple_files = True
)

# A Placeholder to collect all documents

all_docs =[]

if uploaded_files:
    # show progress spinner while loading
    with st.spinner(" 🔄 Loading & Splitting PDFs "):
        for pdf in uploaded_files:
            # write to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix= ".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name

                # Load the PDF into a list of document objects
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
    # split docs into chunks for embedding            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )            
splits = text_splitter.split_documents(all_docs)

# Build or Load the chroma vector store (Caching For Performance)

@st.cache_resource(show_spinner=False)
def get_vectorstore(_splits):
    return Chroma.from_documents(
        _splits,
        embeddings,
        persist_directory= "./chroma_index"
    )
vectorstore = get_vectorstore(splits)
retriever = vectorstore.as_retriever()

# Build a history-aware retriever that uses past chat to refine searched.

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given The Chat History And The Lastest User Question, Decide What To Retrieve."),
    MessagesPlaceholder("chat_history")
    ("human", "{input}"),
])

create_history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)


# QA chain "stuff" all retrieved docs into the LLM

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. Use the retrieved context to answer."
                "if you don't know, say so. Keep it under three sentences. \n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),            
])
question_answer-chain = create_stuff_documents_chain(llm,qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Session state for chat history

if "chathistory" not in st.session_state:
    st.session_state.chathistory={}

def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key = "input",
    history_message_key = "chat_history",
    output_messages_key = "answer",
)   

# Chat UI

session_id = st.text_input("🆔 Session_ID", value= "default_session")
user_question = st.chat_input("✍🏻 Your Question Here...")

if user_question:
    history = get_history(session_id)
    result = conversational_rag.invoke(
        {"input" : user_question},
        config = {"configurable" : {"session_id" : session_id}},
    )
    asnwer = result["answer"]

    # Display in streamlit new chat format
    st.chat_message("user").write(user_question)
    st.chat_message("assistant").write(answer)

    with st.expander("📖 Full chat history"):
        for msg in history.messages:
            # msg role is typically "human" or "assistant"
            role = getattr(msg, "role", msg.type)
            content = msg.content
            st.write(f"** {role.title()}: ** {content}")
else:
    # No file is uploaded yet
    
    st.info("ℹ️ Upload One or more PDFs above to begin.")





