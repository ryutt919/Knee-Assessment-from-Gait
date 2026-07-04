import json
import subprocess
import os

PARENT_PAGE_ID = "32e32e0c-7dbe-8051-b618-d691b63486f4"
TABLE_NOTION_PATH = "/Users/ryutt/Desktop/mini_ryutt/Walking/Study-review/logs/notion_export/06_table_notion.md"
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
    print("Reading Notion table markup...")
    with open(TABLE_NOTION_PATH, "r", encoding="utf-8") as f:
        table_content = f.read().strip()

    client = MCPClient()
    try:
        print("Appending table to the parent page...")
        insert_args = {
            "page_id": PARENT_PAGE_ID,
            "type": "insert_content",
            "insert_content": {
                "content": "\n\n" + table_content,
                "position": {
                    "type": "end"
                }
            }
        }
        resp = client.call_tool("API-update-page-markdown", insert_args)
        if "error" in resp.get("result", {}):
            print(f"Error: {resp['result']['error']}")
            return

        print("Successfully appended table to parent page!")
        print(f"Response: {resp.get('result', {}).get('content', [{}])[0].get('text', 'Success')[:300]}...")

    finally:
        client.close()

if __name__ == "__main__":
    main()
