"""Minimal OpenAI-compatible stub for the local_vcc docker integration test.

Implements just enough of ``POST /v1/chat/completions`` for the VlmClient:
based on ``STUB_VERDICT`` it returns a JSON body matching the
``CoTStructuredOutput`` schema (``{"description": ..., "final_answer": ...}``)
so the worker's structured-output response path is exercised end-to-end
without needing a real GPU.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


VERDICT = os.environ.get("STUB_VERDICT", "yes")  # yes | no | na
PORT = int(os.environ.get("STUB_PORT", "7999"))


class Handler(BaseHTTPRequestHandler):
    def _write(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._write(200, {"status": "ok"})
        else:
            self._write(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)  # ignore body
        if VERDICT.lower() == "na":
            self._write(500, {"error": "stub NA"})
            return
        final_answer = VERDICT.lower() == "yes"
        description = "yes, positively" if final_answer else "no way"
        content = json.dumps({"description": description, "final_answer": final_answer})
        self._write(200, {
            "id": "stub",
            "object": "chat.completion",
            "model": "stub",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
        })

    def log_message(self, fmt, *args):  # silence
        return


if __name__ == "__main__":
    print(f"[stub_vlm] listening on 0.0.0.0:{PORT} verdict={VERDICT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

