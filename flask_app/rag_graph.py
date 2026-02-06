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
LLM_MODEL_NAME = "gpt-4o"  # Default
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)
elif os.getenv("GROQ_API_KEY"):
    # Fallback to Llama 3 on Groq if OpenAI not available
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
else:
    # If neither, we might fail or use a dummy.
    # For now assume keys are present as verified.
    raise ValueError("OPENAI_API_KEY or GROQ_API_KEY required for Advanced RAG.")

# --- State Definition ---
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query_analysis: Literal["generate", "web_search", "retrieve"]
    retrieved_docs: List[any] # Stores Document objects
    grader_result: str
    rewrite_query: str
    active_query: str
    cache_hit: bool

# --- 1. Query Analysis / Routing ---
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["generate", "web_search", "retrieve"] = Field(
        ...,
        description="Given a user question choose to route it to web search, vectorstore retrieval, or general generation."
    )

def route_query_analysis(state: GraphState):
    print("--- ROUTE QUERY ---")
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

# --- 2. Contextualize (History Awareness) ---
def contextualize_query(state: GraphState):
    print("--- CONTEXTUALIZE ---")
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

# --- 3. Retrieval ---
# We need a way to pass the retriever. Since the graph is compiled once, 
# but user retrievers are dynamic (based on user_id), we might need to inject it 
# or use a global lookup if possible. 
# BEST PRACTICE for Multi-Tenant: Pass the retriever or vectorstore in the `config` 
# or construct it dynamically inside the node using user_id from config.

def retrieve(state: GraphState, config: RunnableConfig):
    print("--- RETRIEVE ---")
    query = state["active_query"]
    
    # Dynamic Retrieval Logic
    # We expect 'retriever' to be passed in config['configurable'] or we rebuild it here.
    # Rebuilding is safer for strict isolation.
    user_id = config["configurable"].get("user_id")
    if not user_id:
        # Fallback empty
        return {"retrieved_docs": []}

    # Imports from app logic (avoid circular import if possible, or repeat logic)
    # We will repeat logic slightly or import helper if moved to common.
    # For now, let's assume we can get retrieval function from utils or pass it.
    # Let's try to grab the function from `context` or `configurable`.
    retrieval_fn = config["configurable"].get("retrieval_fn")
    
    docs = []
    if retrieval_fn:
        docs = retrieval_fn(query)
    
    return {"retrieved_docs": docs}

# --- 4. Grader (Self-Correction) ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grader(state: GraphState):
    print("--- CHECK RELEVANCE ---")
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
    
    # We grade essentially if ANY doc is relevant or if the SET is relevant.
    # Simplification: Concatenate key parts or grade top 1. 
    # Let's grade the first few.
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

# --- 5. Query Rewrite ---
def rewrite_query(state: GraphState):
    print("--- REWRITE QUERY ---")
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

# --- 6. Web Search ---
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

# --- 7. Generate ---
def generate(state: GraphState):
    print("--- GENERATE ---")
    question = state["active_query"]
    docs = state.get("retrieved_docs", [])
    
    # Format context
    context = "\n\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for ScholarSync. Use the following context to answer the user's question. If you don't know, say so."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {"messages": [response]}


# --- Graph Construction ---
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

workflow.add_edge("rewrite_query", "retrieve") # Loop back to retrieve with better query
workflow.add_edge("generate", END)

# Compile
rag_graph = workflow.compile()
