import os
import torch
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
import logging
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
import glob
from langchain.evaluation import load_evaluator
from pprint import pprint as print
from langchain import VectorDBQA
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import streamlit as st
from past.builtins import xrange
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} 
bge_embed = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def question_answer(question):
    template = """You are an expert question-answering bot designed to provide in-depth responses based on the given documents.
    Your task is to give detailed response and steps. Again I am telling it should be 9000 words. Don't add "Based on the given document" in the response.
    You should not add this line in answer:
    Based on Given document.
        DOCUMENTS:
        ===========
        {context}
        ===========
        QUESTION: {human_input}
        Chat History:
        {chat_history}
        """

    embeddings=bge_embed
    db=FAISS.load_local('persist_directory_merge/',embeddings)
    prompt_template=PromptTemplate(input_variables=["context", "human_input"], template=template)
    memory = ConversationBufferMemory(input_key='human_input', memory_key="chat_history", return_messages=True)
    chain = load_qa_chain(
            llm=ChatOpenAI(
                openai_api_key=open_ai_key,  # Replace with your OpenAI API key
                model_name="gpt-3.5-turbo",
                # model_kwargs={'presence_penalty': 0.8},
                temperature=0,
                max_tokens=1500
            ),
            chain_type="stuff",
            memory=memory,
            prompt=prompt_template,
        )

       
    
    docs = db.similarity_search(question,k=3)
    set_llm_cache(InMemoryCache())
    #print("Chunks :   ", docs)
    # Assuming documents is the list of Document objects you provided

    response = chain({"input_documents": docs, "human_input": question})
    output_text = response['output_text']
    return output_text
    #print("Response :", output_text)



# Set page title and description
st.title("Chatbot Question-Answering App")
st.write("Ask questions, and the chatbot will provide detailed answers!")

# Get user input
open_ai_key=st.text_input("Enter your open ai key")
question = st.text_input("Ask a question:")

if question:
    # Call the question-answering function
    result = question_answer(question)

    # Display the result
    st.subheader("Chatbot Answer:")
    st.write(result)

# Optionally, you can add more information or instructions to the user
st.markdown("""
    **Instructions:**
    Enter your question in the text box above, and the chatbot will generate a detailed answer.
    """)

# Add any other Streamlit configurations or UI elements as needed
