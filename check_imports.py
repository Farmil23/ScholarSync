
print("Starting import check...")
try:        
    from flask_app.rag_graph import rag_graph
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
