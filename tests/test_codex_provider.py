import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from work.providers.codex import CodexProvider


FAKE_SERVER = r'''
import json
import sys

thread_id = "fake-thread"
turn_id = "fake-turn"

for line in sys.stdin:
    message = json.loads(line)
    request_id = message.get("id")
    method = message.get("method")

    if method == "initialize":
        print(json.dumps({"jsonrpc":"2.0","id":request_id,"result":{"serverInfo":{"name":"fake"}}}), flush=True)
        continue
    if method == "initialized":
        continue
    if method == "thread/start":
        print(json.dumps({"jsonrpc":"2.0","id":request_id,"result":{"thread":{"id":thread_id}}}), flush=True)
        continue
    if method == "thread/resume":
        print(json.dumps({"jsonrpc":"2.0","id":request_id,"result":{"thread":{"id":message["params"]["threadId"]}}}), flush=True)
        continue
    if method == "turn/start":
        print(json.dumps({"jsonrpc":"2.0","id":request_id,"result":{"turn":{"id":turn_id,"status":"inProgress"}}}), flush=True)
        print(json.dumps({"jsonrpc":"2.0","method":"turn/started","params":{"turn":{"id":turn_id}}}), flush=True)
        print(json.dumps({"jsonrpc":"2.0","method":"item/agentMessage/delta","params":{"delta":"Hello from fake Codex"}}), flush=True)
        print(json.dumps({"jsonrpc":"2.0","method":"turn/completed","params":{"turn":{"id":turn_id,"status":"completed"}}}), flush=True)
        continue
'''


class CodexProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.server_path = Path(self.temp_dir.name) / "fake_server.py"
        self.server_path.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")
        self.provider = CodexProvider(command=[sys.executable, str(self.server_path)])
        self.provider.start()

    def tearDown(self):
        self.provider.stop()
        self.temp_dir.cleanup()

    def test_starts_thread_and_streams_turn_to_completion(self):
        thread = self.provider.create_thread(cwd=self.temp_dir.name)
        self.assertEqual(thread["id"], "fake-thread")

        notifications = list(self.provider.send("Build the feature"))
        methods = [notification.get("method") for notification in notifications]

        self.assertIn("turn/started", methods)
        self.assertIn("item/agentMessage/delta", methods)
        self.assertEqual(methods[-1], "turn/completed")
        self.assertEqual(self.provider.turn_id, "fake-turn")

    def test_provider_uses_existing_thread_for_follow_up_turns(self):
        list(self.provider.send("First turn"))
        self.assertEqual(self.provider.thread_id, "fake-thread")
        list(self.provider.send("Second turn"))
        self.assertEqual(self.provider.thread_id, "fake-thread")


if __name__ == "__main__":
    unittest.main()
