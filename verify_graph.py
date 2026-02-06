
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from flask_app.rag_graph import rag_graph

load_dotenv()

# Mock retrieval function for testing
def mock_retrieval(query):
    print(f"DEBUG: Retrieved docs for '{query}'")
    from langchain_core.documents import Document
    return [Document(page_content="ScholarSync is an AI tool to help students with their thesis (skripsi). It uses Advanced RAG.", metadata={"source": "manual.pdf"})]

def run_verification():
    print("--- STARTING VERIFICATION ---")
    
    user_input = "What is ScholarSync?"
    
    # Config with mock retrieval
    config = {"configurable": {"user_id": "test_user", "retrieval_fn": mock_retrieval}}
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "active_query": user_input
    }
    
    print(f"Input: {user_input}")
    
    try:
        # Run graph
        response = rag_graph.invoke(initial_state, config=config)
        
        print("\n--- RESULT ---")
        final_msg = response["messages"][-1].content
        print(f"AI: {final_msg}")
        
        print("\n--- STATE ---")
        print(f"Analysis: {response.get('query_analysis')}")
        print(f"Retrieval Count: {len(response.get('retrieved_docs', []))}")
        print(f"Rewrite: {response.get('rewrite_query')}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
