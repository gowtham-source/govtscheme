import os
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.llms import Ollama
from langchain.tools import DuckDuckGoSearchRun
# from trend import get_trends
from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama


search_tool = DuckDuckGoSearchRun()

load_dotenv()
# # To Load Local models through Ollama
# mistral = Ollama(model="mistral")

# To Load GPT-4
# api = os.environ.get("OPENAI_API_KEY")

# To load gemini (this api is for free: https://makersuite.google.com/app/apikey)
# api_gemini = os.environ.get("GOOGLE_API_KEY")
api_gemini = 'AIzaSyB_LJCLlpqiEYDnJkTQkueLDh0mDnhZ8oo'
# llm = ChatGoogleGenerativeAI(
#     model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
# )

# llm = Ollama(model="gemma:2b")

llm_lmstudio = ChatOpenAI(
    # openai_api_base="http://192.168.137.1:1234/v1",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="",                 
    model_name="mistral"
)


def start_chat(query):
# trend_data = get_trends()
    # query = 'I like to start a startup in india, is there any beneficial scheme available for me'

    researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory="""You are an expert at a technology research group, 
    skilled in identifying new government schemes and analyzing the users complex background.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm_lmstudio
    )
    writer = Agent(
        role='Senior Govt. consultant',
        goal='Craft compelling content on Government Schemes',
        backstory="""You are a content strategist known for 
        making complex Government consulting solutions making it interesting and easy to understand.""",
        verbose=True,
        allow_delegation=True,
        llm=llm_lmstudio
    )

    # Create tasks for your agents
    task1 = Task(
    description=f"""Analyze the following User query {query} based on the Indian Govt, Schemes. 
    Provide a detailed report.""",
    agent=researcher
    )

    task2 = Task(
        description=f"""Create a keen understandable explanation based on the query: {query} using your insights. 
        Make it interesting, clear, and suited for easily understandable for the middle peoples. 
        It should be at least 4 lines.""",
        agent=writer
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[researcher,writer],
        tasks=[task1,task2],
        verbose=2
    )

    # Get your crew to work!
    result = crew.kickoff()

    # print("######################")
    # print(result)

    # with open('source.txt', 'w', encoding='utf-8') as f:
    #     f.write(result)
    return result

query = 'I like to start a startup in india, is there any beneficial scheme available for me'
print(start_chat(query))