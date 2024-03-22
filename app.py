import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import tempfile

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    data = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,
                                                                           convert_system_message_to_human=True),
                                                  retriever=vectors.as_retriever())
    st.header("Customer Support ChatBot")

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk about your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            if "thank you" in user_input.lower():
                response = "You're welcome!"
            elif "how many records" in user_input.lower():
                response = f"There are {len(data)} records in the file."
            else:
                response = conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(response)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
