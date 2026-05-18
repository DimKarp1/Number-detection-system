from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
import json


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        print("\n" + "=" * 80)
        print(datetime.now().isoformat())
        print(f"Path: {self.path}")
        print("Headers:")

        for key, value in self.headers.items():
            print(f"  {key}: {value}")

        print("Body:")

        try:
            parsed = json.loads(body.decode("utf-8"))
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        except Exception:
            print(body.decode("utf-8", errors="replace"))

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 9100), Handler)
    print("Callback receiver listening on http://127.0.0.1:9100/callback")
    server.serve_forever()
