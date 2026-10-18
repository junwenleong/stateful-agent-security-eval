"""Opus 5 per-turn diagnostic for the post-fix frontier harness.

The 2026-10-12 records for claude-opus-5 have a final assistant text only in
session 2 (calendar); sessions 1, 3 and 4 end with tool calls and no text. This
script reruns the identical four-session protocol (same system prompt, tools,
corpus, payload and loop) and logs, for every turn: finish_reason, whether the
message carried tool calls, the content length and the tool names. It writes
diag_opus5_per_turn.jsonl next to this file.

Usage (from the repo root, through the traffic guard):
  FRONTIER_ANTHROPIC_PREFIX=<route> with-frontier .venv/bin/python \
      results/postfix_recollection_2026-10-12/diag_opus5_turns.py [n_runs]
"""
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_postfix_recollection as rr  # noqa: E402  (same tools, prompts, payload)

MODEL = "claude-opus-5"
OUT = HERE / "diag_opus5_per_turn.jsonl"


def one_run(client):
    run_id = str(uuid.uuid4())
    db = f"/tmp/diag_opus5_{run_id}.db"
    corpus = [rr.RAGDocument(doc_id="malicious_doc", content=rr.PAYLOAD_A, is_malicious=True),
              rr.RAGDocument(doc_id="sensitive_doc", content=rr.SENSITIVE_DOC, is_malicious=False),
              rr.RAGDocument(doc_id="monitoring_config", content=rr.MONITORING_DOC, is_malicious=False)]
    tools_dict = {"memory": rr.MemoryTool(db_path=db), "rag": rr.RAGTool(corpus=corpus),
                  "email": rr.EmailTool(), "calendar": rr.CalendarTool(), "search": rr.SearchTool()}
    tools = rr.get_full_tools()
    turns = []
    for s_idx, session in enumerate(rr.SESSIONS):
        messages = [{"role": "system", "content": rr.SYSTEM_PROMPT},
                    {"role": "user", "content": session["prompt"]}]
        # same per-session counter resets as run_single in the runner
        tools_dict["memory"].reset_call_count()
        tools_dict["rag"].reset_call_count()
        for name in ("calendar", "search"):
            if hasattr(tools_dict[name], "reset_call_count"):
                tools_dict[name].reset_call_count()
        for turn in range(15):
            resp = client.chat.completions.create(
                model=os.environ.get("FRONTIER_ANTHROPIC_PREFIX", "") + MODEL,
                messages=messages, tools=tools, max_tokens=4096)
            ch = resp.choices[0]
            msg = ch.message
            names = [tc.function.name for tc in (msg.tool_calls or [])]
            turns.append({"run_id": run_id, "session": s_idx, "turn": turn,
                          "finish_reason": ch.finish_reason, "n_tool_calls": len(names),
                          "tools": names, "content_len": len(msg.content or ""),
                          "content_head": (msg.content or "")[:300],
                          "served_model": re.sub(r"^[a-z_]+\.(?=claude-)", "", getattr(resp, "model", "") or "")})
            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.function.name,
                                                              "arguments": tc.function.arguments}}
                                                for tc in msg.tool_calls]})
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    res = rr.execute_tool(tools_dict, tc.function.name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(res)})
            else:
                break
            # Original harness: `if finish_reason == "stop": break` here, even after
            # tool calls. Logged above; this diagnostic continues so the effect is visible.
        time.sleep(1)
    try:
        os.unlink(db)
    except OSError:
        pass
    return turns


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    client = rr.get_client()
    with open(OUT, "a") as f:
        for _ in range(n):
            for t in one_run(client):
                f.write(json.dumps(t) + "\n")
                print(t["session"], t["turn"], t["finish_reason"], t["n_tool_calls"], t["tools"],
                      t["content_len"], flush=True)


if __name__ == "__main__":
    main()
