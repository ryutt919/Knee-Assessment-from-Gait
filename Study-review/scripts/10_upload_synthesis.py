import json
import subprocess
import os
import time

PARENT_PAGE_ID = "32e32e0c-7dbe-8051-b618-d691b63486f4"
SYNTHESIS_NOTION_PATH = "/Users/ryutt/Desktop/mini_ryutt/Walking/Study-review/logs/notion_export/04_synthesis_notion.md"
import os
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "YOUR_NOTION_TOKEN_HERE")

class MCPClient:
    def __init__(self):
        env = os.environ.copy()
        env["NOTION_TOKEN"] = NOTION_TOKEN
        self.proc = subprocess.Popen(
            ["npx", "-y", "@notionhq/notion-mcp-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        self.msg_id = 1
        self._init_handshake()

    def _send_msg(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _read_msg(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    def _init_handshake(self):
        self._send_msg({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "python-mcp-client",
                    "version": "1.0.0"
                }
            },
            "id": self.msg_id
        })
        self._read_msg()
        self.msg_id += 1

        self._send_msg({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })

    def call_tool(self, name, arguments):
        self._send_msg({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            },
            "id": self.msg_id
        })
        resp = self._read_msg()
        self.msg_id += 1
        return resp

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()

def main():
    print("Reading synthesis markdown...")
    with open(SYNTHESIS_NOTION_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # The H1 title is on the first line
    lines = content.split("\n")
    title = lines[0].replace("#", "").strip()
    body_content = "\n".join(lines[1:]).strip()

    client = MCPClient()
    try:
        print("Creating subpage in Notion...")
        create_args = {
            "parent": {
                "type": "page_id",
                "page_id": PARENT_PAGE_ID
            },
            "properties": {
                "title": {
                    "title": [{"text": {"content": title}}]
                }
            }
        }
        resp = client.call_tool("API-post-page", create_args)
        if "error" in resp.get("result", {}):
            print(f"Error: {resp['result']['error']}")
            return

        result_content = resp["result"]["content"][0]["text"]
        page_data = json.loads(result_content)
        page_id = page_data["id"]
        page_url = page_data["url"]
        print(f"Created subpage {page_id} -> {page_url}")

        print("Updating subpage markdown content...")
        update_args = {
            "page_id": page_id,
            "type": "replace_content",
            "replace_content": {
                "new_str": body_content
            }
        }
        update_resp = client.call_tool("API-update-page-markdown", update_args)
        print(f"Update status: {update_resp.get('result', {}).get('content', [{}])[0].get('text', 'Success')}")

    finally:
        client.close()

if __name__ == "__main__":
    main()
