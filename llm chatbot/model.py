import os
from flask import Flask, request, jsonify, render_template
import random
from mykey import key
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bs4 import BeautifulSoup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = key
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from huggingface_hub import notebook_login
from langchain.llms import HuggingFacePipeline
#from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
#from langchain.memory import ConversationBufferMemory
from langdetect import detect
from translate import Translator
import wikipediaapi

DB_FAISS_PATH = 'myVectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, provide a general response.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


app = Flask(__name__)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

    return prompt

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_llm():
    
    # Model ID
    repo_id = 'meta-llama/Llama-2-7b-chat-hf'

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        device_map='auto',
        load_in_4bit=True
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=True
    )

    # Create pipeline
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    # Load the LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


@app.route('/')
def index():
    greeting = "Welcome to the Chatbot App! How can I assist you today?"
    return render_template('chatbot.html', greeting=greeting)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        query = data['query']
        
        if query.lower() in ["hi", "hello", "hey","Is anyone there?","good day","what's up","heyy","how are you","whatsupp"]:
            responses = ["Hello!","Good to see you again!","Hi there, how can I help?","Hello, How can I help you?","Assalamualikum"]
            response = random.choice(responses)
            #response = "Hello! How can I help you?"
        elif query.lower() in ["bye", "goodbye", "end"]:
            response = "Goodbye! If you have more questions, feel free to ask later."
                
        else:

                print("Asking from LLM")
                qa_result = qa_bot()
                response = qa_result({'query': query})

                if 'result' in response:
                    response = response['result']
                else:
                    response = "No answer found"

        return jsonify({'answer': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

