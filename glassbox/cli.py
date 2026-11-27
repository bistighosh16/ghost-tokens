"""
Ghost Tokens CLI — Command line interface.
Run: ghost-tokens serve
Made with 🔮 by Vivi
"""
import sys
import argparse
from pathlib import Path


def serve_command(args):
    """Launch the FastAPI server and web visualizer."""
    import uvicorn
    
    # Ensure project root is in sys.path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        
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
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="ghost-tokens",
        description="Ghost Tokens - See where model confidence gets thin",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # ghost-tokens serve
    serve_parser = subparsers.add_parser("serve", help="Start visualizer web server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()