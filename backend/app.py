import pinecone
import os
import openai
import pickle
import tempfile
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA

# from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

"Flask packages"
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


# pinecone
api_key = os.environ.get("PINECONE_API_KEY")
env = os.environ.get("PINECONE_ENV")

# Setting up ability to query Pinecone with tools

# Vectorstore Index
index_name = "podcasts"

# Pincone API key

api_key = "API_KEY"
env = "asia-southeast1-gcp-free"
openapi_config = OpenApiConfiguration.get_default_copy()
openapi_config.proxy = "http://proxy.server:3128"
memory = None
app = Flask(__name__)
CORS(app)


@app.route("/api/execute-gpt-query/<string:namespace>", methods=["POST"])
@cross_origin()
def execute_python_function(namespace):
    OPENAI_API_KEY=""
    default_prompt = """You are OrbitAI, a friendly search engine built by Kaiser who provides informative answers to users. Answer the following questions as best you can. You have access to the following tools:"""
    new_suffix = """Begin! Remember to give detailed, informative answers. If possible, try to directly enter the question into the tool.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""
    openai.api_key=OPENAI_API_KEY
    llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)
    openapi_config = OpenApiConfiguration.get_default_copy()
    
    pinecone.init(api_key=api_key, environment=env)
    global memory
    if memory is None:
        try:
            with open("memory.pickle", "rb") as f:
                memory = pickle.load(f)
        except FileNotFoundError:
            memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(temperature=0, openai_api_key=openai.api_key ),memory_key="chat_history", return_messages=True, max_token_limit=2000)


    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_existing_index(
        index_name, embeddings, namespace=namespace
    )

    embeddings_retriever = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    # podcast_retriever = RetrievalQAWithSourcesChain.from_chain_type(llm=retrieval_llm, chain_type="stuff", retriever=docsearch.as_retriever())

    tools = [
        Tool(
            name = 'Kaiser Cheng Infomation Search',
            func=embeddings_retriever.run,
            description=" use this search of text when you need additional information about Kaiser Cheng. Input should be a fully formed question.",

        )
    ]


    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, max_token_limit=500, max_iterations=4, agent_kwargs={'PREFIX': default_prompt, 'SUFFIX': new_suffix})

    print(request.json)
    input_data = request.json.get("input")
    print(input_data)

    try:
         output = agent_chain.run(input=input_data)
    except Exception as e:
         output = str(e)
         if not output.startswith("Could not parse LLM output:"):
             raise e
         output = output.removeprefix("Could not parse LLM output: ").removesuffix("`")
    

    with open("memory.pickle", "wb") as f:
        pickle.dump(memory, f)
    return jsonify({"output": output})


@app.route("/upload/<string:namespace>", methods=["POST"])
@cross_origin()
def upload_file(namespace):
    if "file" not in request.files:
        return "No file received", 401

    file = request.files["file"]

    if file.filename == "":
        return "No file selected", 402

    # Save the uploaded PDF file to a temporary location
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    print(file.filename)
    file.save(file_path)
    filename_without_ext, _ = os.path.splitext(file.filename)

    # Process the uploaded PDF file
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key
                                 )

    pinecone.init(
        api_key="PINECONE_API_KEY",
        environment="asia-southeast1-gcp-free",
        openapi_config=openapi_config
    )

    index_name = "podcasts"

    # Create a Pinecone object and index the PDF content
    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name, namespace=filename_without_ext)
    docsearch = Pinecone.from_documents(
        docs, embeddings, index_name=index_name, namespace=namespace
    )

    return "File uploaded and processed successfully, inserted into pinecone"

@app.route("/namespacevalidity/<string:namespace>", methods=["POST"])
@cross_origin()
def check_namespace_validity(namespace):
    api_key = "721f43b4-3f81-4e8d-b42e-2a0eb86e5141"
    env = "asia-southeast1-gcp-free"
    pinecone.init(api_key=api_key, environment=env, openapi_config=openapi_config)
    index = pinecone.Index(index_name)
    index_stats_response = index.describe_index_stats()
    if namespace in index_stats_response.namespaces.keys():
        return '1'
    return '0'

@app.route("/api/delete-memory", methods=["DELETE"])
@cross_origin()

def delete_memory():
    global memory

    if not os.path.exists("memory.pickle"):
        return jsonify({"message": "Memory file not found."})

    try:
        os.remove("memory.pickle")
        memory = None  # Set memory to None
        return jsonify({"message": "Memory file deleted successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
