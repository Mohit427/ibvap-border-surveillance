"""Single entry point.

Local use: `python run.py` starts the server and opens the dashboard in
your browser at http://localhost:8000

Container/cloud use (see Dockerfile): reads HOST/PORT/browser behavior from
env vars so the same script works unchanged in a Hugging Face Space.
"""
import os
import threading
import time
import webbrowser

import uvicorn

HOST = os.environ.get("IBVAP_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", os.environ.get("IBVAP_PORT", "8000")))
OPEN_BROWSER = os.environ.get("IBVAP_OPEN_BROWSER", "1") == "1"


def _open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://{'localhost' if HOST == '0.0.0.0' else HOST}:{PORT}")


if __name__ == "__main__":
    if OPEN_BROWSER:
        threading.Thread(target=_open_browser, daemon=True).start()
    print(f"\nIBVAP starting -> http://{HOST}:{PORT}\n(Ctrl+C to stop)\n")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, log_level="info")
