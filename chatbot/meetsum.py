import json
import os
import time

import joblib
import nest_asyncio
# from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
import nltk
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import (JSONLoader, PyPDFLoader,
                                                  TextLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama, OpenAI
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from streamlit.components.v1 import html
from streamlit_chat import message

load_dotenv()
# api_key = "952c8164-6ad6-44c2-b9c1-236a20a63eb7"
openai_api_key = "ssk-proj-1cMQ0pDQ0hUsCzTrLP2r6rjE-A2lpNlWwxTBxluZ0r0ujfCxTXx3Hs8j2-RFGB5fJggzmgn4WpT3BlbkFJU1VXe03RPf-F1mxmN0CQblIO-EYtAEcGSwJC0KX11cI0GBVzA3Ah1j8IGUSuqNBWM600tx6cwA"
# api_key= os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key= os.environ.get("PINECONE_API_KEY"))


api_key = "gbJWBLPOrzwjFAM4pTamxSmSnRmCoC3W3VTSVdpM"
embeddings = CohereEmbeddings(cohere_api_key=api_key , model="embed-english-v3.0")
template = """<s>[INST] Given the context - {context} </s>[INST] [INST] Answer the following question - {question}[/INST]

If someone greets you with hello or hi , reply in greeting form and nicely .
If the context is unclear, kindly provide more information or rephrase the question. If the topic is outside my expertise, I'll let you know.

"""

pt = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

nest_asyncio.apply()
# vectorDB storage
vectorStore = PineconeVectorStore(index_name='rag-chatmodel', embedding=embeddings)



rag_mistral = RetrievalQA.from_chain_type(
            llm=Ollama(model="mistral"),
            retriever=vectorStore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 3},
            ),
            memory=ConversationSummaryMemory(llm=Ollama(model="mistral")),
            chain_type_kwargs={"prompt": pt, "verbose": True},
        )





if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

st.set_page_config(layout="wide")
col1, col2 = st.columns([1,2])
app = Flask(__name__)


def create_pkl_string(filename):
    file_name, extension = os.path.splitext(filename)
    new_string = file_name + ".pkl"
    return new_string



lemmatizer = WordNetLemmatizer()

porter = PorterStemmer()
import string


def transform_text(text) : 
    text = text.lower()
    text = nltk.word_tokenize(text)  #the text has come in the list  
 
    y=[]
    for i in text: 
        if i.isalnum(): #special character are removed
            y.append(i)

    text = y[:] 
    y.clear() 
    for i in text : 
        if i not in stopwords.words('english') and i not in string.punctuation : 
            y.append(i)

    text = y[:]
    y.clear()
    for i in text : 
        y.append(porter.stem(i.lower())) 



    return " ".join(y)



llamaparse_api_key = "llx-8MMHGFCJ5PKqyfZM6h5D8epMtjzG4OEOe6lMCEOvgu67YgIt"
def load_or_parse_data(file_name):
    # data_file = "data/Introduction-of-MS-Office-MS-Word-PDF-eng.pkl"

    changed_file_ext = create_pkl_string(file_name)
    print("changed_file_ext", changed_file_ext)
    data_file = f"data/{changed_file_ext}"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionUber10k = """The provided document is unstructured
        It contains many tables, text, image and list.
        Try to be precise while answering the questions"""
        parser = LlamaParse(
            api_key=llamaparse_api_key,
            result_type="markdown",
            parsing_instruction=parsingInstructionUber10k,
            max_timeout=5000,
        )
        llama_parse_documents = parser.load_data(f"src/{file_name}")
        # llama_parse_documents = parser.load_data("data/Introduction-of-MS-Office-MS-Word-PDF-eng.pdf")
        print("llama_parse_documents", llama_parse_documents)
        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, f"data/{file_name}")

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data





def create_vector_database(file_name):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data(file_name)
    print("llama_parse_documents", llama_parse_documents[0].text[:300])


    with open("data/output.md", "w", encoding="utf-8") as f:
        for doc in llama_parse_documents:
            f.write(doc.text + "\n")

    markdown_path = "data/output.md"
    print("markdown_path", markdown_path)
    loader = UnstructuredMarkdownLoader(markdown_path, encoding="utf-8")


    documents = loader.load()

    print("documents", documents)
    print(f"length of documents loaded: {len(documents)}")
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    

    docs = text_splitter.split_documents(documents)
    # split_docs = splitter.split_documents(docs)

    for i in range(len(docs)) : 
        docs[i].page_content = transform_text(docs[i].page_content)
       
       
    vectorStore = PineconeVectorStore.from_documents(
        docs, embeddings, index_name="rag-chatmodel"
    )

    len(docs)
    print("docs loaded:", docs)
    print(f"length of docs loaded: {len(docs)}")
    print(f"total number of document chunks generated :{len(docs)}")
    docs[0]

    # Prepare texts and metadatas
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # Inserting to index
    PineconeVectorStore.from_texts(
        texts, embeddings, index_name="rag-chatmodel", metadatas=metadatas
    )

    

    print("Vector DB created successfully !")
    return


def reRanker():
    # compressor = FlashrankRerank()

    retriever = vectorStore.as_retriever(search_kwargs={"k": 10})
    # retriever.search_type

    # print("retriever.search_typ", retriever.search_type)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorStore.as_retriever(
            search_kwargs={"k": 10},
        ),
    )
    print("compression_retrieverInside", compression_retriever)

    return compression_retriever

def convert_to_vector(file_name):
    global vectorStore
    global embeddings
    try:
        folder_path = "./src"
        file_path = os.path.join(folder_path, file_name)

        # Check if the file exists in the folder
        if not os.path.exists(file_path):
            return {"error": "File not found in the server"}

        index_name = "rag-chatmodel"

        # delete the old data
        existing_indexes = pc.list_indexes()
        if existing_indexes and existing_indexes[0].name == index_name:
            pc.delete_index(index_name)

        # Create a new data
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        # Initialize the appropriate loader based on file type
        if file_name.lower().endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_name.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_name.lower().endswith('.csv'):
            loader = CSVLoader(file_path)
        elif file_name.lower().endswith('.json'):
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='[]',
                text_content=False)
        elif file_name.lower().endswith('.docx'):
            # Example: Handle DOCX files using PyPDFLoader or another appropriate loader
            loader = PyPDFLoader(file_path)
        else:
            return {"error": "Unsupported file format"}
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        

        create_vector_database(file_name)
        


        # docs = loader.load()
        
        

        
        

        return {"message": "File processed and vectors added successfully"}

    except Exception as e:
        error_message = str(e)
        app.logger.error(f"Error in upload API: {error_message}")
        return {"error": error_message}



def upload():
    try:

        if not st.session_state.get("files"):
            return {"error": "No files uploaded"}

        file = st.session_state["files"][0]
        if file.name == '':
            return {"error": "No selected file"}

        folder_path = "./src"
        file_path = os.path.join(folder_path, file.name)

        # Check if the file already exists in the folder
        if os.path.exists(file_path):
            return {"message": "File already exists"}

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        return {"message": "File uploaded successfully"}
    except Exception as e:
        error_message = str(e)
        app.logger.error(f"Error in upload API: {error_message}")
        return {"error": error_message}

def chat_llama(question):
    try:
        global vectorStore
        if vectorStore is None:
            return {"error": "Vector store not initialized"}

        if not question:
            return {"error": "No question provided"}
        
        rag = RetrievalQA.from_chain_type(
            llm=Ollama(model="llama3"),
            retriever=vectorStore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5,"k": 1}),
            memory=ConversationSummaryMemory(llm = Ollama(model="llama3")),
            chain_type_kwargs={"prompt": pt, "verbose": True},
        )


        response = rag.invoke(question)
        
        # Check if response is a dictionary
        if isinstance(response, dict):
            # Access values directly from the dictionary
            if 'result' in response:
                result = response['result']
            else:
                result = None
            if 'history' in response:
                history = response['history']
            else:
                history = None
                
            st.session_state.prompts.append(question)
            if result:
                st.session_state.responses.append(result)
            # else:
            print("TTTTTTt",result )
            return result
        else:
            return {"error": "Invalid response format"}

    except Exception as e:
        error_msg = f"Error in chat API: {str(e)}"
        app.logger.error(error_msg)  # Log the error to a file
        return {"error": error_msg}

    
    

def chat_gpt(question):

    try:
    
        global vectorStore
        if vectorStore is None:
            return {"error": "Vector store not initialized"}

        if not question:
            return {"error": "No question provided"}
        

        rag = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                api_key=openai_api_key[1:],
                temperature=0,
                model="gpt-4o",
            ),
            retriever=vectorStore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 3},
            ),
            memory=ConversationSummaryMemory(
                llm=ChatOpenAI(
                    api_key=openai_api_key[1:],
                    temperature=0,
                    model="gpt-4o",
                )
            ),
            chain_type_kwargs={"prompt": pt, "verbose": True},
        )

        response = rag.invoke(question)
        print("This is the resposne" , response)
        
        if isinstance(response, dict):
            # Access values directly from the dictionary
            if 'result' in response:
                result = response['result']
            else:
                result = None
            if 'history' in response:
                history = response['history']
            else:
                history = None
                
            st.session_state.prompts.append(question)
            if result:
                st.session_state.responses.append(result)
            # else:
            print("TTTTTTt",result )
            return result
        else:
            return {"error": "Invalid response format"}

    except Exception as e:
        error_msg = f"Error in chat API: {str(e)}"
        app.logger.error(error_msg)  # Log the error to a file
        return {"error": error_msg}


# chat_gpt api
# @app.route("/chat_gpt", methods=["POST"])
def chat_mistral(question):
    try:
        global vectorStore
        if vectorStore is None:
            return {"error": "Vector store not initialized"}

        if not question:
            return {"error": "No question provided"}

        rag = RetrievalQA.from_chain_type(
            llm=Ollama(model="mistral"),
            retriever=vectorStore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5,"k": 1}),
            memory=ConversationSummaryMemory(llm = Ollama(model="mistral")),
            chain_type_kwargs={"prompt": pt, "verbose": True},
        )

        response = rag.invoke(question)
        print(response)
        
        # Check if response is a dictionary
        if isinstance(response, dict):
            # Access values directly from the dictionary
            if 'result' in response:
                result = response['result']
            else:
                result = None
            if 'history' in response:
                history = response['history']
            else:
                history = None
                
            st.session_state.prompts.append(question)
            if result:
                st.session_state.responses.append(result)
            # else:
            print("TTTTTTt",result )
            return result
        else:
            return {"error": "Invalid response format"}

    except Exception as e:
        error_msg = f"Error in chat API: {str(e)}"
        app.logger.error(error_msg)  # Log the error to a file
        return {"error": error_msg}





chat_option = None 


with st.sidebar : 
    
    if 'converted_file' not in st.session_state:
        st.session_state.converted_file = None

    st.title("MeetSum-chatbot")
    
    # st.title("Upload your docs")
    chat_option = st.sidebar.radio("Select Assistant:", ("llama3" ,"Mistral", "GPT"))
    
    if 'prev_chat_option' not in st.session_state:
        st.session_state.prev_chat_option = None 

    if chat_option != st.session_state.prev_chat_option:
        st.session_state.prompts = []  # Clear the prompts list
        st.session_state.responses = []  # Clear the responses list
        st.session_state.prev_chat_option = chat_option



if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.processing = False  # Add a processing flag

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

prompt = st.chat_input("Ask your question", disabled=st.session_state.processing)



if prompt and not st.session_state.processing:
    st.session_state.processing = True  # Set the processing flag
    try:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
                if (prompt != None) :
                    prompt = transform_text(prompt)

            if chat_option == "Mistral":
                response = chat_mistral(prompt)
                result = chat_mistral(prompt)
            elif chat_option == "GPT":
                response = chat_gpt(prompt)
                result = chat_gpt(prompt)
            elif chat_option == "llama3":
                response = chat_llama(prompt)
                result = chat_llama(prompt)

            # result = chat_gpt(prompt)
            if result:
                st.session_state.messages.append({"role": "assistant", "content": result})
                with st.chat_message("assistant"):
                    st.write(result)
            else:
                st.session_state.messages.append({"role": "assistant", "content": "No response"})
                with st.chat_message("assistant"):
                    st.write("No response")
    finally:
        st.session_state.processing = False









