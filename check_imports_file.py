
import sys

with open("debug_log.txt", "w") as f:
    f.write("Starting import check...\n")
    f.flush()
    
    try:
        from flask_app.rag_graph import rag_graph
        f.write("Import successful!\n")
    except Exception as e:
        f.write(f"Import failed: {e}\n")
        import traceback
        traceback.print_exc(file=f)
    f.flush()
