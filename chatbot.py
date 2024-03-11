import os
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from flask_cors import CORS
from langchain.llms import Ollama
from dotenv import load_dotenv
from data_preparation import load_data

load_dotenv()
os.environ['OPENAI_API_KEY'] = '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'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# llm_lmstudio = ChatOpenAI(
#     # openai_api_base="http://192.168.137.1:1234/v1",
#     openai_api_base="http://localhost:1234/v1",
#     openai_api_key="",                 
#     model_name="mistral",
#     temperature=0
# )

retriever = load_data()

api_gemini = 'AIzaSyB_LJCLlpqiEYDnJkTQkueLDh0mDnhZ8oo'
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)

template = """[Role]: As a government scheme consultant Chatbot, I specialize in assisting middle-income individuals in understanding and accessing appropriate government schemes.If it is a general question, Ignore the above context and reply normally
Contexts:
{context}

Question: {question}
"""

template2 = """
You are a government scheme consultant chatbot, who is specialized in assisting individuals in understanding  and accessing appropriate government schemes,. The user  asks you a question and a context will be provided, If the context is based on the question answer accordingly otherwise respond as a polite assistant. If the context is not related to the question do not generate anything on the context.
CONTEXT:
{context}

QUESTION:
{question}

Reponse:
"""
prompt = ChatPromptTemplate.from_template(template2)

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

class Question(BaseModel):
    __root__: str

chain = chain.with_types(input_type=Question)

# query = input('User: ')
# result = chain.invoke(query)
# print(result)

@app.route("/", methods=["GET"])
def index():
    return "Hello, World!"

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_input = data['query']
    result = chain.invoke(user_input)
    return jsonify({'response': result})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000, host="0.0.0.0")