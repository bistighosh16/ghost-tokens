"""
Ghost Tokens Server Runner — Launch FastAPI backend and web visualizer.
Made with 🔮 by Vivi
"""
import argparse
import sys
from pathlib import Path
import uvicorn

# Ensure the root directory is on the Python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Run the Ghost Tokens web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    print("=" * 60)
    print("  Ghost Tokens visualizer")
    print(f"  Serving at: http://{args.host}:{args.port}")
    print("  Built on GlassBox")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "glassbox.server.app:app",
        host=args.host,
        port=args.port,
        reload=False,  # Single process mode prevents Windows spawn path bugs
    )


if __name__ == "__main__":
    main()