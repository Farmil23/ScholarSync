import os
from typing import Annotated, Literal, TypedDict, List
from dotenv import load_dotenv

# LangChain / LangGraph Imports
# LangChain / LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq

# from langchain_community.vectorstores import FAISS # REMOVED: Too heavy
# from langchain_community.docstore.in_memory import InMemoryDocstore # REMOVED
# from langchain_community.tools.tavily_search import TavilySearchResults # REMOVED: Too heavy dependency
from langchain_core.tools import Tool

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
# import faiss # REMOVED: Too heavy


# Load Env
load_dotenv()

# --- Configuration ---
# Prioritize OpenAI for advanced reasoning, fallback to Groq if OpenAI missing but Groq present.
# Global variable for cached graph
# Global variable for cached graph
_cached_graph = None
_cached_llm_reasoning = None
_cached_llm_gen = None

def get_llm(streaming=False):
    global _cached_llm_reasoning, _cached_llm_gen
    
    # Check cache
    if streaming and _cached_llm_gen:
        return _cached_llm_gen
    if not streaming and _cached_llm_reasoning:
        return _cached_llm_reasoning

    # Check keys lazily
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if openai_key:
        # GPT-4o for generation, GPT-3.5-turbo or 4o-mini for reasoning could be cheaper, but stick to 4o for now.
        llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=streaming)
    elif groq_key:
        llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, streaming=streaming)
    else:
        raise ValueError("Configuration Error: OPENAI_API_KEY or GROQ_API_KEY is missing.")
        
    if streaming:
        _cached_llm_gen = llm
        return llm
    else:
        _cached_llm_reasoning = llm
        return llm

# --- Node Functions ---

def route_query_analysis(state: GraphState):
    print("--- ROUTE QUERY ---")
    llm = get_llm(streaming=False) # No Stream
    question = state.get("active_query") or state["messages"][-1].content
    
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system_prompt = "You are an expert router. Choose 'retrieve', 'web_search', or 'generate'."
    route_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    router = route_prompt | structured_llm_router
    
    try:
        decision = router.invoke({"question": question}).datasource
    except:
        decision = "generate"
        
    print(f"Decision: {decision}")
    return {"query_analysis": decision, "loop_step": 0} # Reset loop step

def contextualize_query(state: GraphState):
    print("--- CONTEXTUALIZE ---")
    llm = get_llm(streaming=False) # No Stream
    messages = state["messages"]
    if len(messages) <= 1:
         return {"active_query": messages[-1].content}
         
    system_prompt = "Reformulate the latest question to be standalone given the history."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"active_query": chain.invoke({"chat_history": messages[:-1], "question": messages[-1].content})}

def retrieve(state: GraphState, config: RunnableConfig):
    print("--- RETRIEVE ---")
    query = state["active_query"]
    user_id = config["configurable"].get("user_id")
    retrieval_fn = config["configurable"].get("retrieval_fn")
    docs = retrieval_fn(query) if (user_id and retrieval_fn) else []
    return {"retrieved_docs": docs}

def grader(state: GraphState):
    print("--- CHECK RELEVANCE ---")
    llm = get_llm(streaming=False) # No Stream
    question = state["active_query"]
    docs = state["retrieved_docs"]
    loop_step = state.get("loop_step", 0)
    
    if loop_step >= 3:
        print("--- LOOP LIMIT REACHED: FORCE GENERATE ---")
        return {"grader_result": "yes", "rewrite_query": "no"}
    
    if not docs:
        return {"grader_result": "no", "rewrite_query": "yes"}
        
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system_prompt = "Grade relevance 'yes' or 'no'."
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Doc: {document} \n Question: {question}"),
    ])
    grader_chain = grade_prompt | structured_llm_grader
    
    for d in docs[:3]:
        content = d.page_content if hasattr(d, 'page_content') else str(d)
        if grader_chain.invoke({"question": question, "document": content}).binary_score == "yes":
             return {"grader_result": "yes", "rewrite_query": "no"}
    
    return {"grader_result": "no", "rewrite_query": "yes"}

def rewrite_query(state: GraphState):
    print("--- REWRITE QUERY ---")
    llm = get_llm(streaming=False) # No Stream
    question = state["active_query"]
    loop_step = state.get("loop_step", 0)
    
    msg = [HumanMessage(content=f"Rewrite this question to be better for vector search: {question}")]
    response = llm.invoke(msg)
    return {"active_query": response.content, "loop_step": loop_step + 1}

def web_search(state: GraphState):
    print("--- WEB SEARCH ---")
    query = state["active_query"]
    tavily_key = os.getenv("TAVILY_API_KEY")
    docs = []
    
    if tavily_key:
        try:
            import requests
            res = requests.post("https://api.tavily.com/search", json={"query": query, "api_key": tavily_key, "max_results": 3}).json()
            docs = [f"Source: {d['url']}\nContent: {d['content']}" for d in res.get("results", [])]
        except: docs = []
    else:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                docs = [f"Source: {r['href']}\nContent: {r['body']}" for r in list(ddgs.text(query, max_results=3))]
        except: docs = ["Web search unavailable."]
    return {"retrieved_docs": docs}

def generate(state: GraphState, config: RunnableConfig):
    print("--- GENERATE ---")
    # ONLY THIS NODE USES STREAMING LLM
    llm = get_llm(streaming=True) 
    question = state["active_query"]
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for ScholarSync. Answer the question using the context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    
    # We pass 'config' here so the QueueCallback gets attached to THIS chain only
    response = (prompt | llm).invoke({"context": context, "question": question}, config=config)
    return {"messages": [response]}


def get_rag_graph():
    global _cached_graph
    if _cached_graph: return _cached_graph
    get_llm(streaming=False) # Pre-check keys
    
    workflow = StateGraph(GraphState)
    workflow.add_node("contextualize", contextualize_query)
    workflow.add_node("query_analysis", route_query_analysis)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grader", grader)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "query_analysis")
    
    workflow.add_conditional_edges("query_analysis", lambda x: x["query_analysis"], {"web_search": "web_search", "retrieve": "retrieve", "generate": "generate"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grader")
    workflow.add_conditional_edges("grader", lambda x: "rewrite" if x["rewrite_query"] == "yes" else "generate", {"rewrite": "rewrite_query", "generate": "generate"})
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    
    _cached_graph = workflow.compile()
    return _cached_graph

