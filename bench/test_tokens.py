# -*- coding: utf-8 -*-
"""token 记账相关的自测（不联网、不碰真实数据）：老库升级、并发累加、截断/缺 usage 的计数。

跑法：.venv/Scripts/python.exe bench/test_tokens.py
"""
import json
import shutil
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ok = True


def check(name, cond, detail=""):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  {detail}" if detail else ""))
    ok = ok and bool(cond)


# ---------- 1. 老库（没有 tokens 列）升级后数据不丢 ----------
print("1) 老库自动升级")
from app import config, db  # noqa: E402

tmp = Path(tempfile.mkdtemp())
old = tmp / "old.db"
c = sqlite3.connect(old)
c.executescript("""
CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL DEFAULT 'link',
  aweme_id TEXT UNIQUE, url TEXT, local_path TEXT, title TEXT, author TEXT, duration REAL, cover TEXT,
  status TEXT NOT NULL DEFAULT 'queued', reason TEXT, attempts INTEGER NOT NULL DEFAULT 0,
  transcript TEXT, note_json TEXT, category TEXT, md_path TEXT, learned INTEGER NOT NULL DEFAULT 0,
  starred INTEGER NOT NULL DEFAULT 0, my_note TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
  updated_at REAL NOT NULL, learned_at REAL, segments_json TEXT, deep_json TEXT, chat_json TEXT,
  media_type TEXT, images_json TEXT);
""")
c.execute("INSERT INTO items(title,transcript,status,created_at,updated_at) VALUES('旧数据','文字稿','done',1,1)")
c.commit()
c.close()
db.DB_PATH = old
db.init()
cols = {r[1] for r in db.conn().execute("PRAGMA table_info(items)")}
check("补上 tokens 列", "tokens" in cols)
row = db.get(1)
check("原数据还在", row and row["title"] == "旧数据" and row["status"] == "done", f"tokens={row['tokens']}")
db.update(1, tokens=4321)
check("能写能读", db.get(1)["tokens"] == 4321)
db._local.conn.close()

# ---------- 2. 并发累加不丢数 ----------
print("2) 并发记账")
bad = 0
for r in range(20):
    config.SETTINGS_PATH = tmp / f"s{r}.json"
    config.save({"tokens_used": 0, "tokens_weekly": {}})

    def w():
        for _ in range(20):
            config.add_tokens(100)

    ts = [threading.Thread(target=w) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))["tokens_used"] != 8000:
        bad += 1
check("4 线程 x 20 次 x 100 = 8000，20 轮全部对上", bad == 0, f"错 {bad} 轮")

# ---------- 3. 截断、缺 usage 时怎么算 ----------
print("3) 请求层的计数")
from app import llm  # noqa: E402

cfg = {"kind": "openai", "api_key": "x", "base_url": "https://example.com/v1", "model": "m"}
calls = []


def fake_post(url, headers, payload, timeout=120):
    calls.append(payload.get("max_tokens") or payload.get("max_completion_tokens"))
    if len(calls) == 1:  # 第一次被截断，花掉 9000
        return {"choices": [{"message": {"content": '{"points": ["a"]'}, "finish_reason": "length"}],
                "usage": {"total_tokens": 9000}}
    return {"choices": [{"message": {"content": '{"points": ["【标题】正文"]}'}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 500}}


llm._post = fake_post
llm._MAX_OUT.clear()
text, used = llm.chat(cfg, [{"role": "user", "content": "讲点什么"}], max_tokens=1000)
check("截断那次的 9000 也计入", used == 9500, f"used={used}，两次上限={calls}")

# 缺 usage 时按字数兜底
llm._post = lambda *a, **k: {"choices": [{"message": {"content": "中" * 1000}, "finish_reason": "stop"}]}
_, used = llm.chat(cfg, [{"role": "user", "content": "字" * 1000}], max_tokens=100)
check("没 usage 时估算而不是记 0", 1100 <= used <= 1300, f"used={used}（2000 字）")

shutil.rmtree(tmp, ignore_errors=True)
print("\n全部通过" if ok else "\n有失败项")
sys.exit(0 if ok else 1)
