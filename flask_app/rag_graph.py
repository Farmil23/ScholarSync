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
_cached_graph = None
_cached_llm = None

def get_llm():
    global _cached_llm
    if _cached_llm:
        return _cached_llm
        
    # Prioritize OpenAI for advanced reasoning, fallback to Groq if OpenAI missing but Groq present.
    # Check keys lazily
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if openai_key:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    elif groq_key:
        llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, streaming=True)
    else:
        raise ValueError("Configuration Error: OPENAI_API_KEY or GROQ_API_KEY is missing in environment variables.")
        
    _cached_llm = llm
    return llm

# --- Node Functions (Now call get_llm()) ---

def route_query_analysis(state: GraphState):
    print("--- ROUTE QUERY ---")
    llm = get_llm()
    question = state.get("active_query") or state["messages"][-1].content
    
    # Simple structured router
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system_prompt = """You are an expert router. 
    1. 'retrieve': If the user asks about specific documents, files, candidates, CVs, or uploaded content (skripsi, tesis, etc).
    2. 'web_search': If the user asks current events, news, or general facts NOT in the documents.
    3. 'generate': If the user asks simple greetings, general chitchat, or definitions not requiring search.
    """
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    router = route_prompt | structured_llm_router
    try:
        result = router.invoke({"question": question})
        decision = result.datasource
    except:
        decision = "generate" # Default fallback
        
    print(f"Decision: {decision}")
    return {"query_analysis": decision}

def contextualize_query(state: GraphState):
    print("--- CONTEXTUALIZE ---")
    llm = get_llm()
    messages = state["messages"]
    if len(messages) <= 1:
         return {"active_query": messages[-1].content}
         
    system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed, otherwise return it as is."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    history = messages[:-1]
    question = messages[-1].content
    
    refined = chain.invoke({"chat_history": history, "question": question})
    print(f"Refined Query: {refined}")
    return {"active_query": refined}

def retrieve(state: GraphState, config: RunnableConfig):
    print("--- RETRIEVE ---")
    query = state["active_query"]
    
    # Dynamic Retrieval Logic
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return {"retrieved_docs": []}

    retrieval_fn = config["configurable"].get("retrieval_fn")
    
    docs = []
    if retrieval_fn:
        docs = retrieval_fn(query)
    
    return {"retrieved_docs": docs}

def grader(state: GraphState):
    print("--- CHECK RELEVANCE ---")
    llm = get_llm()
    question = state["active_query"]
    docs = state["retrieved_docs"]
    
    if not docs:
        return {"grader_result": "no", "rewrite_query": "yes"}
        
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    grader_chain = grade_prompt | structured_llm_grader
    
    relevant_found = False
    filtered_docs = []
    
    for d in docs[:3]: # check top 3
        # Handle if doc is string or Document object
        content = d.page_content if hasattr(d, 'page_content') else str(d)
        score = grader_chain.invoke({"question": question, "document": content})
        if score.binary_score == "yes":
            relevant_found = True
            filtered_docs.append(d)
            
    if relevant_found:
         return {"grader_result": "yes", "rewrite_query": "no", "retrieved_docs": filtered_docs}
    
    return {"grader_result": "no", "rewrite_query": "yes"}

def rewrite_query(state: GraphState):
    print("--- REWRITE QUERY ---")
    llm = get_llm()
    question = state["active_query"]
    
    msg = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent / meaning. \n 
            Initial Question: {question} \n 
            Formulate an improved question: """,
        )
    ]
    model = llm
    response = model.invoke(msg)
    return {"active_query": response.content}

def web_search(state: GraphState):
    print("--- WEB SEARCH ---")
    query = state["active_query"]
    
    docs = []
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_key:
        try:
            # Lightweight Tavily Client (No LangChain Community needed)
            import requests
            response = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "api_key": tavily_key, "max_results": 3}
            )
            results = response.json().get("results", [])
            formatted = [f"Source: {d['url']}\nContent: {d['content']}" for d in results]
            processed_docs = formatted
        except Exception as e:
            print(f"Tavily Error: {e}")
            processed_docs = []
    else:
        # Fallback DDGS
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                processed_docs = [f"Source: {r['href']}\nContent: {r['body']}" for r in results]
        except ImportError:
             processed_docs = ["Web search unavailable (ddgs module missing)."]
            
    return {"retrieved_docs": processed_docs}

def generate(state: GraphState, config: RunnableConfig):
    print("--- GENERATE ---")
    llm = get_llm()
    question = state["active_query"]
    docs = state.get("retrieved_docs", [])
    
    # Format context
    context = "\n\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for ScholarSync. Use the following context to answer the user's question. If you don't know, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question}, config=config)
    
    return {"messages": [response]}


def get_rag_graph():
    global _cached_graph
    if _cached_graph:
        return _cached_graph
        
    # Ensure LLM is ready (will raise error if keys missing)
    get_llm()
    
    # Graph Construction
    workflow = StateGraph(GraphState)
    
    # Nodes
    workflow.add_node("contextualize", contextualize_query)
    workflow.add_node("query_analysis", route_query_analysis)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grader", grader)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    
    # Edges
    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "query_analysis")
    
    def route_decision(state):
        return state["query_analysis"]
    
    workflow.add_conditional_edges(
        "query_analysis",
        route_decision,
        {
            "web_search": "web_search",
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grader")
    
    def grader_decision(state):
        return state["rewrite_query"]
    
    workflow.add_conditional_edges(
        "grader",
        lambda x: "rewrite" if x["rewrite_query"] == "yes" else "generate",
        {
            "rewrite": "rewrite_query",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("rewrite_query", "retrieve") 
    workflow.add_edge("generate", END)
    
    # Compile
    _cached_graph = workflow.compile()
    return _cached_graph

