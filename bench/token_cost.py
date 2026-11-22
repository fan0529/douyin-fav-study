# -*- coding: utf-8 -*-
"""量一条笔记真实消耗多少 token：用同一条视频的转录，按 llm.extract 完全相同的
提示词和参数发一次请求，读接口返回的 usage 明细（含推理 token）。
用法：.venv/Scripts/python.exe bench/token_cost.py [item_id]
"""
import json
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import llm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
item_id = int(sys.argv[1]) if len(sys.argv) > 1 else 3

s = json.load(open(os.path.join(ROOT, "data", "settings.json"), encoding="utf-8"))
con = sqlite3.connect(os.path.join(ROOT, "data", "library.db"))
con.row_factory = sqlite3.Row
it = dict(con.execute("select * from items where id=?", (item_id,)).fetchone())

cats = "、".join(s.get("categories") or [])
system = llm.SYSTEM.replace("{focus}", s.get("focus_topic") or "不限").replace("{categories}", cats)
user = llm.USER.format(title=it["title"] or "（无）", author=it["author"] or "（未知）",
                       label="语音转文字稿", transcript=(it["transcript"] or "")[:llm.MAX_CHARS])

print(f"视频：{it['title'][:24]}…")
print(f"时长：{it['duration']/60:.1f} 分钟   转录：{len(it['transcript'])} 字")
print(f"system：{len(system)} 字   user：{len(user)} 字   合计：{len(system)+len(user)} 字")

payload = {
    "model": s["model"],
    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    "temperature": 0.3,
    "stream": False,
    "max_tokens": llm.NOTE_MAX_TOKENS,
    "reasoning_effort": s.get("reasoning_effort") or "low",
    "response_format": {"type": "json_object"},
}
req = urllib.request.Request(s["base_url"].rstrip("/") + "/chat/completions",
                             data=json.dumps(payload).encode("utf-8"), method="POST",
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {s['api_key']}"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.loads(r.read().decode("utf-8"))
dt = time.time() - t0

u = data.get("usage") or {}
print("\n===== usage 原文 =====")
print(json.dumps(u, ensure_ascii=False, indent=2))
text = data["choices"][0]["message"]["content"] or ""
print(f"\n耗时 {dt:.0f}s   finish_reason={data['choices'][0].get('finish_reason')}")
print(f"输出正文 {len(text)} 字")

pt = u.get("prompt_tokens") or 0
ct = u.get("completion_tokens") or 0
tot = u.get("total_tokens") or (pt + ct)
rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or u.get("reasoning_tokens") or 0
cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("cached_tokens") or 0
print("\n===== 汇总 =====")
print(f"输入 prompt_tokens      {pt}")
print(f"输出 completion_tokens  {ct}（其中思考 {rt}）")
print(f"合计 total_tokens       {tot}")
print(f"命中缓存 cached_tokens  {cached}")
