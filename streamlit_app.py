# How to load multiple PDFs into vector Database
# How to split large documents into small chunks for better embeddings
# how to use retrieval augmented generations (RAG) with langchain chain that combines a vector store retriever + an llm (GROQ) + Embedding (Hugging Face) + a prompt template + conversational Q&A Chat + Unique Session ID Wise 
# process from upload till extraction
# Load PDF file -> convert their contents into vector embeddings -> implemented a chat history so that each conversation is remembered -> how the user session logic (session_ID) helps each user to maintain their own conversation flow
# PDF Q&A with RAG, Embeddings, Chat History, and Session Handling

import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Streamlit config
st.set_page_config(
    page_title="\ud83d\udcc4 RAG Q&A with PDF & Chat History",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("\ud83d\udcc4 RAG Q&A WITH PDF UPLOADS & CHAT HISTORY")
st.sidebar.header("\ud83d\udc69\u200d\ud83d\udcbb Configuration")
st.sidebar.write("""
- Enter Your GROQ API Key 
- Upload PDFs On The Main Page 
- Ask Questions And See Chat History
""")

# API key input
api_key = st.sidebar.text_input("Groq API Key", type="password")

# Stop if no API key
if not api_key:
    st.warning("\ud83d\udd11Please Enter Your GROQ API Key In The Sidebar To Continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Upload PDFs
uploaded_files = st.file_uploader("\ud83d\udcc1 Choose PDF Files(s)", type="pdf", accept_multiple_files=True)
all_docs = []

if uploaded_files:
    with st.spinner("\ud83d\udd04 Loading & Splitting PDFs"):
        for pdf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)

    @st.cache_resource(show_spinner=False)
    def get_vectorstore(_splits):
        return Chroma.from_documents(_splits, embeddings, persist_directory="./chroma_index")

    vectorstore = get_vectorstore(splits)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given The Chat History And The Lastest User Question, Decide What To Retrieve."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Use the retrieved context to answer. If you don't know, say so. Keep it under three sentences.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if "chathistory" not in st.session_state:
        st.session_state.chathistory = {}

    def get_history(session_id: str):
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        return st.session_state.chathistory[session_id]

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_message_key="chat_history",
        output_messages_key="answer",
    )

    session_id = st.text_input("\ud83c\udd94 Session_ID", value="default_session")
    user_question = st.chat_input("\u270d\ufe0f Your Question Here...")

    if user_question:
        history = get_history(session_id)
        result = conversational_rag.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}},
        )
        answer = result["answer"]

        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)

        with st.expander("\ud83d\udcd6 Full chat history"):
            for msg in history.messages:
                role = getattr(msg, "role", msg.type)
                st.write(f"**{role.title()}:** {msg.content}")
else:
    st.info("\u2139\ufe0f Upload One or more PDFs above to begin.")





