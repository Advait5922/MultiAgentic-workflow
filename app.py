import streamlit as st
from dotenv import load_dotenv
import os
from typing import Annotated, Sequence, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.riza.command import ExecPython
from langchain_groq import ChatGroq
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from pprint import pprint

# Load environment variables
load_dotenv()

# Retrieve API keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
RIZA_API_KEY = os.getenv('RIZA_API_KEY')

# Set environment variables
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["RIZA_API_KEY"] = RIZA_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize the ChatGroq object
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# Initialize tools
tool_tavily = TavilySearchResults(max_results=2)
tool_code_interpreter = ExecPython()
tools = [tool_tavily, tool_code_interpreter]

# System prompt for the supervisor
system_prompt = ('''You are a workflow supervisor managing a team of three agents: Prompt Enhancer, Researcher, and Coder. Your role is to direct the flow of tasks by selecting the next agent based on the current stage of the workflow. For each task, provide a clear rationale for your choice, ensuring that the workflow progresses logically, efficiently, and toward a timely completion.

**Team Members**:
1. Enhancer: Use prompt enhancer as the first preference, to Focuse on clarifying vague or incomplete user queries, improving their quality, and ensuring they are well-defined before further processing.
2. Researcher: Specializes in gathering information.
3. Coder: Handles technical tasks related to calculation, coding, data analysis, and problem-solving, ensuring the correct implementation of solutions.

**Responsibilities**:
1. Carefully review each user request and evaluate agent responses for relevance and completeness.
2. Continuously route tasks to the next best-suited agent if needed.
3. Ensure the workflow progresses efficiently, without terminating until the task is fully resolved.

Your goal is to maximize accuracy and effectiveness by leveraging each agent’s unique expertise while ensuring smooth workflow execution.
''')

# Define the Supervisor class
class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Specifies the next worker in the pipeline: "
                    "'enhancer' for enhancing the user prompt if it is unclear or vague, "
                    "'researcher' for additional information gathering, "
                    "'coder' for solving technical or code-related problems."
    )
    reason: str = Field(
        description="The reason for the decision, providing context on why a particular worker was chosen."
    )

# Define the supervisor node function
def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    response = llm.with_structured_output(Supervisor).invoke(messages)
    goto = response.next
    reason = response.reason
    print(f"Current Node:  Supervisor -> Goto: {goto}")
    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,
    )

# Define the enhancer node function
def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    system_prompt = (
        "You are an advanced query enhancer. Your task is to:\n"
        "Don't ask anything to the user, select the most appropriate prompt"
        "1. Clarify and refine user inputs.\n"
        "2. Identify any ambiguities in the query.\n"
        "3. Generate a more precise and actionable version of the original request.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    enhanced_query = llm.invoke(messages)
    print(f"Current Node: Prompt Enhancer -> Goto: Supervisor")
    return Command(
        update={
            "messages": [
                HumanMessage(content=enhanced_query.content, name="enhancer")
            ]
        },
        goto="supervisor",
    )

# Define the research node function
def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    research_agent = create_react_agent(
        llm,
        tools=[tool_tavily],
        state_modifier="You are a researcher. Focus on gathering information and generating content. Do not perform any other tasks"
    )
    result = research_agent.invoke(state)
    print(f"Current Node: Researcher -> Goto: Validator")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="validator",
    )

# Define the code node function
def code_node(state: MessagesState) -> Command[Literal["validator"]]:
    code_agent = create_react_agent(
        llm,
        tools=[tool_code_interpreter],
        state_modifier=(
            "You are a coder and analyst. Focus on mathematical caluclations, analyzing, solving math questions, "
            "and executing code. Handle technical problem-solving and data tasks."
        )
    )
    result = code_agent.invoke(state)
    print(f"Current Node: Coder -> Goto: validator")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="validator",
    )

# System prompt providing clear instructions to the validator agent
system_prompt = '''
You are a workflow validator. Your task is to ensure the quality of the workflow. Specifically, you must:
- Review the user's question (the first message in the workflow).
- Review the answer (the last message in the workflow).
- If the answer satisfactorily addresses the question, signal to end the workflow.
- If the answer is inappropriate or incomplete, signal to route back to the supervisor for re-evaluation or further refinement.
Ensure that the question and answer match logically and the workflow can be concluded or continued based on this evaluation.

Routing Guidelines:
1. 'supervisor' Agent: For unclear or vague state messages.
2. Respond with 'FINISH' to end the workflow.
'''

# Define a Validator class for structured output from the LLM
class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

# Define the validator node function
def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    response = llm.with_structured_output(Validator).invoke(messages)
    goto = response.next
    reason = response.reason

    if goto == "FINISH" or goto == END:
        goto = END
        print("Transitioning to END")
    else:
        print(f"Current Node: Validator -> Goto: Supervisor")
    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="validator")
            ]
        },
        goto=goto,
    )

# Initialize the StateGraph
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("enhancer", enhancer_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
builder.add_node("validator", validator_node)

builder.add_edge(START, "supervisor")
graph = builder.compile()

# Streamlit App
st.title("MultiAgent Nexus 2")

# Input for user query
user_input = st.text_input("Enter your query:")

# Button to process the query
if st.button("Process Query"):
    inputs = {
        "messages": [
            ("user", user_input),
        ]
    }

    st.write("Processing your query...")
    final_answer = None
    for output in graph.stream(inputs):
        for key, value in output.items():
            if value is None:
                continue
            st.write(f"Output from node '{key}':")
            st.json(value)

            # Capture the final answer before validation
            if key == "researcher" or key == "coder":
                final_answer = value["messages"][-1].content

    # Display the final answer in a separate section
    if final_answer:
        st.subheader("Final Answer:")
        st.write(final_answer)