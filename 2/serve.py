# -*- coding: utf-8 -*-
"""
Lightweight presentation server for Machine Failure Early-Warning System (EWS).
Runs a local HTTP server and automatically opens the dashboard in your default browser.
"""
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    os.chdir(DIRECTORY)
    url = f"http://localhost:{PORT}/machine_failure_dashboard.html"
    print("=" * 70)
    print("  MACHINE FAILURE EARLY-WARNING SYSTEM: PRESENTATION SERVER")
    print("=" * 70)
    print(f"Serving at: {url}")
    print("Press Ctrl+C to stop the server.\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped cleanly.")
    except Exception as e:
        print(f"Error running server: {e}")

if __name__ == "__main__":
    main()
