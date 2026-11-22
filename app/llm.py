"""AI 提炼：调用 OpenAI 兼容的 /chat/completions 接口，把文字稿变成结构化学习笔记。

只把文字稿发给你选的 AI 服务商；视频、音频、你的个人笔记都不会上传。
"""
import json
import re
import time
import urllib.error
import urllib.request

# 文字稿超过这个长度就截断（只影响发给 AI 的部分，原文转录照常完整保存）。
# 默认「最长转录分钟数」是 15 分钟，中文口播约 4000 字；调到 20000 可以覆盖一小时左右的视频。
MAX_CHARS = 20000

# 单次提炼最多让 AI 输出多少 token。详细版笔记比摘要版长得多，
# 不写这个字段时服务商默认一般只有 4096，JSON 会在中间被截断。
# 注意：推理类模型（GLM-5.x、Qwen3、o 系列等）的「思考」也算在这个额度里，
# 思考能吃掉七八千，所以额度要给得比笔记本身的长度更大。
NOTE_MAX_TOKENS = 16000

# 提炼笔记时让推理模型少思考。实测 GLM-5.3-Flash：
#   默认档 211 秒（思考 11261 token，还容易把额度吃光导致 JSON 截断）
#   low 档 52 秒（思考 9 token，要点条数和质量基本一样）
# 设置里把 reasoning_effort 改成 high/max 可以换回「想得更久」。
NOTE_REASONING = "low"

# 注意：这个模板用的是 str.replace 填占位符，不是 str.format，
# 所以里面的 JSON 花括号可以直接写单括号，不用写成 {{ }}。
SYSTEM = """你是一位「逐段还原型」的学习笔记记录员，不是摘要员。

你会收到一条抖音视频的语音转文字稿，或者一条抖音图文笔记的正文和图片里识别出的文字（都可能有错别字或断行，请按上下文理解），需要整理成一份可以直接照着学的详细笔记。

用户关注的主题：{focus}

━━ 一、总原则：完整还原，不做压缩 ━━

1. 按视频原本的讲述顺序记录。视频怎么讲，笔记就怎么排，不要重组成「三大要点」这种结构。
2. 视频里出现的每一个知识点、每一个例子、每一组数字、每一个操作步骤、每一句「注意」「坑在这里」「我踩过」，都要收进笔记。
3. 不许用概括句代替具体内容。
   反例：「介绍了三种提高效率的方法」。
   正例：「方法一：先把当天要做的事写在纸上，按 25 分钟一段拆开，每段结束休息 5 分钟（番茄工作法）」。
4. 不许把不同段落的内容揉成一条，也不许为了显得简洁而删掉铺垫、类比、失败经验。只有寒暄、口头禅、明显重复的啰嗦可以略过。
5. 视频里说错或前后矛盾的地方，照实记录，并在该条末尾补一句「（注：视频这里说的是 X，实际可能是 Y）」。
6. 转写明显错乱、无法判断原意的片段，写「（此处转写不清，原意可能是……）」，不要凭空补全。

━━ 二、输出格式 ━━

请严格只输出一个 JSON 对象，不要输出任何其他文字，也不要用 ``` 包起来，格式如下：

{
  "skip": false,
  "skip_reason": "",
  "relevant": true,
  "title": "一句话说清这条视频讲了什么（20字以内，不要照抄原标题里的话题标签）",
  "category": "从这些分类里选最合适的一个：{categories}；都不合适可以自拟一个简短分类",
  "tags": ["3-5个关键词"],
  "difficulty": "入门 / 进阶 / 高级 选一个",
  "summary": "80-150 字的全文导览：这条视频完整讲了什么、按什么顺序展开、适合什么基础的人看",
  "points": ["按讲述顺序逐条记录，一般 12-30 条，每条 60-150 字"],
  "tools": [{"name": "视频提到的工具/网站/产品名", "use": "它是干什么用的"}],
  "concepts": [{"term": "一句判断句的结论，20字以内", "explain": "为什么重要、什么场合用得上，1-2句"}],
  "quiz": [{"q": "根据视频内容出的一道自测题", "a": "答案"}]
}

━━ 三、各字段怎么写 ━━

▸ summary：不要写「本视频介绍了……」，直接写内容本身。

▸ points：这是笔记的主体，也是「不删减」的落点。
  · 每条最前面用【】写这一条的小标题，例如：【第一步：选素材】，后面跟具体内容。
  · 每一项都必须是一个字符串，小标题和正文写在同一个字符串里，不要写成 ["小标题", "正文"] 这种数组。
  · 一条只讲一件事。内容多就多分几条，不要全部挤在一条里。
  · 必须保留视频里的具体例子、数字、参数、工具名、价格、失败经验、注意事项。
  · 视频分章节讲的，章节名直接用作小标题；没有章节就按内容自然分段。
  · 视频教了操作方法的，把步骤按实际操作顺序写进 points 里（每条一步或几步），不要另外单独列一份「照着做」的步骤清单——那会和上面的记录重复。

▸ tools：除了名字和用途，视频里提到怎么用、免费还是收费、有什么限制，都写进 use 里。

▸ concepts：这块不是术语表——术语的解释已经在 points 里写过了，不要再重复名词定义，也不要写工具名。
  · 写「通篇读完之后才能得出的结论」，2-4 条。必须是跨越多个段落才成立的判断，不能是某一段的复述。
  · term 写成一句判断句（20 字以内），例如「先搞懂每类技术解决什么问题，再学具体工具」；不要写成「反向代理」这种名词。
  · explain 写为什么重要、在什么场合用得上，1-2 句，40-80 字。
  · 其中一条的 explain 末尾可以补一句「延伸：…」，指出视频没展开、但值得继续了解的方向；没有合适内容就不写。

▸ quiz：考细节和容易理解错的地方，不要考标题和主题。

━━ 四、其他规则 ━━

- 如果视频是广告带货、短剧剧情、唱歌跳舞、纯娱乐闲聊，或几乎没有实质信息，返回 {"skip": true, "skip_reason": "简短原因"}。
- 如果有实质内容，但和用户关注的主题无关，"relevant" 填 false，其他字段照常详细填写。
- 笔记正文一律用纯文本：不要用 **加粗**、## 标题、`代码` 这类 Markdown 标记，需要强调就用【】或「」。数字、英文缩写、专有名词保持原样。
- 不要编造视频里没有的信息；转写里明显听错的专有名词（比如工具名、产品名）可以按常识纠正。
- 全部使用简体中文。quiz 出 3-5 题。
- 宁可写长，不要写短；但内容确实很多时，优先保证 points 写全，quiz 可以减到 3 题。"""

USER = """标题：{title}
作者：{author}
{label}：
{transcript}

请按视频原本的讲述顺序，完整还原成详细笔记：讲到的例子、数字、步骤、注意事项都要保留，不要压缩成概括。"""


class LLMError(Exception):
    def __init__(self, msg, retry=False):
        super().__init__(msg)
        self.retry = retry


def _post(url, headers, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _estimate_used(messages, reply=""):
    """接口没返回 usage 时按字符数估一个，避免把消耗静默记成 0。

    实测中文正文约 0.6 token/字（GLM-5.3-Flash：10794 字 → 6632 token，8539 字 → 5065 token），
    英文和 JSON 符号更省，整体取 0.6 已经够贴近。
    """
    chars = sum(len(str(m.get("content") or "")) for m in messages) + len(reply or "")
    return int(chars * 0.6)


def _as_cfg(c):
    """兼容传整份设置或已经整理好的模型配置。"""
    if "kind" in c:
        return c
    from .config import llm_cfg
    return llm_cfg(c, "main")


_QUIRKS = {}  # (接口地址, 模型) → 这个模型需要的参数调整，记住后下次直接用
_MAX_OUT = {}  # (接口地址, 模型) → 实测的单次输出上限，超过就一定被截断


def _apply_quirks(payload, key):
    q = _QUIRKS.get(key, set())
    if "mct" in q and "max_tokens" in payload:
        # 推理模型的输出上限也包含"思考"用的 token，给宽一点
        payload["max_completion_tokens"] = max(payload.pop("max_tokens") * 2, 4000)
    if "no_temp" in q:
        payload.pop("temperature", None)
    if "no_reason" in q:
        payload.pop("reasoning_effort", None)


def chat(cfg, messages, json_mode=True, temperature=0.3, max_tokens=None, timeout=120, retry_429=True,
         reasoning=None):
    """返回 (文本, 消耗的 token 数)。带重试：网络/5xx 重试 2 次，429/529 限速等 5 秒。

    cfg 可以是整份设置，也可以是 config.llm_cfg() 的结果。
    reasoning：推理模型的思考档位 "low" / "high" / "max"，None 表示交给服务商默认。
    服务商不认这个参数时会自动去掉（有些接口直接 400）。
    """
    cfg = _as_cfg(cfg)
    key = (cfg.get("api_key") or "").strip()
    base = (cfg.get("base_url") or "").strip().rstrip("/")
    model = (cfg.get("model") or "").strip()
    if not key:
        raise LLMError("还没填 AI 的 API Key")
    if not base or not model:
        raise LLMError("请在设置里填好接口地址和模型名")

    if cfg.get("kind") == "anthropic":
        # Claude 原生接口：system 单独传；没有 JSON 模式，靠提示词约束 + parse_json 兜底
        if base.endswith("/v1"):
            base = base[:-3]
        url = base + "/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        payload = {"model": model, "max_tokens": max_tokens or 4096, "messages": msgs, "temperature": temperature}
        if system:
            payload["system"] = system
    else:
        url = base + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if reasoning:
            payload["reasoning_effort"] = reasoning
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        _apply_quirks(payload, (base, model))

    attempt = 0
    trunc_retry = False
    acc = 0  # 被截断重发的那几次也已经计费了，最后一起加上
    while True:
        attempt += 1
        try:
            data = _post(url, headers, payload, timeout=timeout)
            if cfg.get("kind") == "anthropic":
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                u = data.get("usage") or {}
                used = ((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)) or _estimate_used(messages, text)
                if data.get("stop_reason") == "max_tokens" and json_mode:
                    raise LLMError("AI 输出太长被截断了，可以在设置里把精读分段字数调小再试")
            else:
                ch = data["choices"][0]
                text = ch["message"]["content"] or ""
                used = int((data.get("usage") or {}).get("total_tokens") or 0) or _estimate_used(messages, text)
                if ch.get("finish_reason") == "length" and json_mode:
                    # 输出到这个模型的上限了。原因多半是推理模型把额度花在「思考」上，
                    # 正文只写了一半。先试一次更大的上限，不行就把能救的部分救回来。
                    qk = (base, model)
                    cur = payload.get("max_completion_tokens") or payload.get("max_tokens") or 0
                    cap = _MAX_OUT.get(qk)
                    if not trunc_retry and cur < 32000 and (cap is None or cur < cap):
                        trunc_retry = True
                        acc += used
                        nxt = min(max(cur * 2, 8192), 32000)
                        if "max_completion_tokens" in payload:
                            payload["max_completion_tokens"] = nxt
                        else:
                            payload["max_tokens"] = nxt
                        continue
                    fixed = _salvage_json(text)
                    if fixed is not None:
                        print(f"[提醒] AI 输出被截断（上限 {cur} tokens），已按生成到一半的内容保存。",
                              flush=True)
                        return json.dumps(fixed, ensure_ascii=False), used + acc
                    raise LLMError(
                        f"AI 输出被截断：单次最多只能写 {cur} 个 token，这条视频内容太长写不下。"
                        "可以换输出上限更大的模型，或者把「最长转录分钟数」调小一点。")
            return text, used + acc
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "ignore")[:300]
            except Exception:
                pass
            if e.code == 401:
                raise LLMError("API Key 无效（401），请到设置里检查")
            if e.code in (402, 403):
                raise LLMError(f"账户余额不足或没有权限（{e.code}），请到服务商后台检查：{body}")
            if e.code == 404:
                raise LLMError(f"接口地址或模型名不对（404）：{body}")
            low = body.lower()
            qk = (base, model)
            # OpenAI 新一代（推理）模型：max_tokens 改名 max_completion_tokens
            if e.code == 400 and "max_completion_tokens" in low and "max_tokens" in payload:
                _QUIRKS.setdefault(qk, set()).add("mct")
                _apply_quirks(payload, qk)
                continue
            # 有的模型只允许默认 temperature
            if e.code == 400 and "temperature" in low and "temperature" in payload:
                _QUIRKS.setdefault(qk, set()).add("no_temp")
                payload.pop("temperature", None)
                continue
            # 有的接口不认思考档位，去掉重发
            if e.code == 400 and "reasoning_effort" in payload and (
                    "reasoning" in low or "thinking" in low or "思考" in body):
                _QUIRKS.setdefault(qk, set()).add("no_reason")
                payload.pop("reasoning_effort", None)
                continue
            if e.code == 400 and (payload.get("max_tokens") or 0) > 4096 and "token" in low:
                _MAX_OUT[qk] = 4096  # 免费模型通常限制单次最多输出 4096 tokens，记住它
                payload["max_tokens"] = 4096
                continue
            if e.code == 400 and "response_format" in payload:
                payload.pop("response_format")  # 有的模型不支持 JSON 模式，去掉再试
                continue
            if e.code in (429, 529) and attempt <= 3 and retry_429:
                time.sleep(5 * attempt)
                continue
            if e.code >= 500 and attempt <= 2:
                time.sleep(2)
                continue
            if e.code == 429:
                raise LLMError("请求太频繁或免费额度用完了（429）。免费额度一般按天/按月恢复；"
                               "着急的话可以在设置里换成付费模型。", retry=True)
            raise LLMError(f"AI 接口报错（HTTP {e.code}）：{body}", retry=e.code >= 500)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt <= 2:
                time.sleep(2)
                continue
            raise LLMError(f"连不上 AI 接口：{getattr(e, 'reason', e)}", retry=True)
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"AI 接口返回的格式看不懂：{e}")


def parse_json(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise


def _folded(messages):
    """把 system 消息里的任务和格式要求并进第一条 user 消息。

    有的 OpenAI 兼容网关不把 system 消息交给模型，模型只看到材料，就会回一句普通的话。
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system:
        return None
    for m in rest:
        if m["role"] == "user":
            m["content"] = system + "\n\n" + m["content"]
            return rest
    return [{"role": "user", "content": system}] + rest


def _salvage_json(text):
    """把被截断的 JSON 尽量救回来：丢掉最后一条不完整的元素，然后把括号补齐。

    只在输出到上限被截断时用，救不回来返回 None。
    """
    t = (text or "").strip()
    a = t.find("{")
    if a < 0:
        return None
    t = t[a:]
    stack, in_str, esc, cuts = [], False, False, []
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
        elif ch == ",":
            cuts.append((i, tuple(stack)))  # 这里正好在两个元素之间，可以安全截断
    for i, snap in reversed(cuts):
        try:
            d = json.loads(t[:i].rstrip() + "".join(reversed(snap)))
        except Exception:
            continue
        if isinstance(d, dict) and d.get("points"):
            return d
    return None


def chat_json(cfg, messages, **kw):
    """调用 AI 并把回复解析成 JSON，返回 (dict, 消耗的 token 数)。

    回复不是 JSON 时，先把被截断的部分尽量补全；还不行就把 system 里的要求并进
    user 消息再试一次；最后才报错。
    """
    text, used = chat(cfg, messages, **kw)
    try:
        return parse_json(text), used
    except Exception:
        pass
    fixed = _salvage_json(text)
    if fixed is not None:
        return fixed, used
    retry = _folded(messages)
    if retry:
        text, more = chat(cfg, retry, **kw)
        used += more
        try:
            return parse_json(text), used
        except Exception:
            fixed = _salvage_json(text)
            if fixed is not None:
                return fixed, used
    if not (text or "").strip():
        raise LLMError("AI 什么都没返回。多半是输出额度被模型的「思考」占满了："
                       "换一个输出上限更大的模型，或把设置里的 reasoning_effort 改成 low")
    raise LLMError("AI 返回的不是有效的 JSON：" + (text or "")[:200])


def _str_list(v):
    if isinstance(v, str):
        v = [v]
    out = []
    for x in v or []:
        if isinstance(x, (list, tuple)):  # 模型偶尔把一条要点拆成 ["小标题", "正文"]
            x = " ".join(str(i).strip() for i in x if str(i).strip())
        x = str(x).strip()
        if x:
            out.append(x)
    return out


_END_PUNCT = "。！？；：.!?;:，,"


def _merge_headings(points):
    """模型有时把小标题单独放成一条（如「开场：xxx」不带正文），把它并到下一句开头。

    正常写法的每一条都以【小标题】开头、以句号结尾，不受影响。
    """
    out, buf = [], None
    for p in points:
        is_head = len(p) <= 24 and not p.startswith("【") and p[-1] not in _END_PUNCT
        if is_head:
            if buf:
                out.append(buf)
            buf = p
            continue
        out.append(f"【{buf}】{p}" if buf else p)
        buf = None
    if buf:
        out.append(buf)
    return out


def _obj_list(v, keys):
    out = []
    for x in v or []:
        if isinstance(x, dict):
            d = {k: str(x.get(k, "")).strip() for k in keys}
            if d[keys[0]]:
                out.append(d)
        elif isinstance(x, str) and x.strip():
            out.append({keys[0]: x.strip(), keys[1]: ""})
    return out


def normalize(d: dict) -> dict:
    """统一字段格式，校验不过就抛错。"""
    if d.get("skip") is True:
        return {"skip": True, "skip_reason": str(d.get("skip_reason") or "没有实质内容")}
    note = {
        "skip": False,
        "relevant": d.get("relevant") is not False,
        "title": str(d.get("title") or "").strip(),
        "category": str(d.get("category") or "其他").strip() or "其他",
        "tags": _str_list(d.get("tags"))[:6],
        "difficulty": str(d.get("difficulty") or "").strip(),
        "summary": str(d.get("summary") or "").strip(),
        "points": _merge_headings(_str_list(d.get("points"))),
        "steps": _str_list(d.get("steps")),
        "tools": _obj_list(d.get("tools"), ["name", "use"]),
        "concepts": _obj_list(d.get("concepts"), ["term", "explain"]),
        "quiz": _obj_list(d.get("quiz"), ["q", "a"]),
    }
    if not note["points"]:
        raise LLMError("AI 返回的核心观点为空")
    if not note["title"]:
        note["title"] = note["points"][0][:30]
    return note


def _reasoning_level(settings):
    """思考档位：设置里写 low / high / max；写 none / off / default 表示交给服务商。"""
    v = str(settings.get("reasoning_effort") or NOTE_REASONING).strip().lower()
    return None if v in ("", "none", "off", "default", "auto") else v


def extract(settings, title, author, transcript, kind="video"):
    """返回 (笔记 dict, token 数)。"""
    cats = "、".join(settings.get("categories") or [])
    # 用 replace 填占位符（不是 format）：SYSTEM 里有 JSON 花括号，format 会把它们当占位符。
    system = (SYSTEM
              .replace("{focus}", settings.get("focus_topic") or "不限")
              .replace("{categories}", cats))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": USER.format(
            title=title or "（无）", author=author or "（未知）",
            label="图文笔记的正文和图片文字（【图N】是第 N 张图里识别出的文字）" if kind == "note" else "语音转文字稿",
            transcript=(transcript or "")[:MAX_CHARS])},
    ]
    d, used = chat_json(settings, messages, max_tokens=NOTE_MAX_TOKENS,
                        reasoning=_reasoning_level(settings), timeout=600)
    note = normalize(d)
    if not note.get("skip"):
        # AI 有时把「AI 编程」写成「AI编程」，同一个分类就被拆成两个；对上设置里的分类名
        same = {re.sub(r"\s+", "", c): c for c in settings.get("categories") or []}
        note["category"] = same.get(re.sub(r"\s+", "", note["category"]), note["category"])
    return note, used


CHAT_HINT = ("gpt", "o1", "o3", "o4", "chatgpt", "claude", "deepseek", "qwen", "glm", "moonshot", "kimi",
             "yi-", "llama", "mistral", "gemini", "doubao", "ernie", "hunyuan", "minimax", "abab", "step")
SKIP_HINT = ("embed", "whisper", "tts", "dall-e", "image", "audio", "realtime", "moderation", "transcribe",
             "search", "rerank", "vision-preview", "babbage", "davinci", "sora", "speech")


def list_models(cfg):
    """向服务商要可用模型列表，返回模型 ID 列表（尽量只留聊天模型）。"""
    cfg = _as_cfg(cfg)
    key = (cfg.get("api_key") or "").strip()
    base = (cfg.get("base_url") or "").strip().rstrip("/")
    if not key or not base:
        raise LLMError("先填好接口地址和 API Key")
    if cfg.get("kind") == "anthropic":
        if base.endswith("/v1"):
            base = base[:-3]
        url, headers = base + "/v1/models?limit=100", {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url, headers = base + "/models", {"Authorization": f"Bearer {key}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"这个服务商不支持读取模型列表（HTTP {e.code}），请手动填写模型名")
    except Exception as e:
        raise LLMError(f"读取模型列表失败：{e}")
    ids = [str(m.get("id")) for m in (data.get("data") or data.get("models") or []) if isinstance(m, dict) and m.get("id")]
    chat_ids = [i for i in ids if not any(h in i.lower() for h in SKIP_HINT)]
    pref = [i for i in chat_ids if any(h in i.lower() for h in CHAT_HINT)]
    out = pref or chat_ids or ids
    return sorted(set(out), key=lambda x: (x.count("-") > 4, x))


def test(cfg, retry_429=True):
    text, _ = chat(cfg, [{"role": "user", "content": "请只回复两个字：你好"}], json_mode=False, temperature=0,
                   max_tokens=300, retry_429=retry_429)
    return text.strip()[:50] or "（连接正常）"


def free_setup(plan, api_key, log=lambda m: None):
    """免费方案一键配置：读模型列表 → 挑免费模型 → 逐个测试，返回 (能用的模型, 试过的列表)。"""
    from .config import FREE_PLANS
    p = FREE_PLANS[plan]
    cfg = {"provider": p["provider"], "base_url": p["base_url"], "api_key": api_key.strip(), "kind": "openai"}
    listed = []
    try:
        listed = [m.split("/", 1)[1] if m.startswith("models/") else m for m in list_models(cfg)]
    except LLMError:
        pass
    cands = [m for m in p["models"] if not listed or m in listed]
    if listed and p.get("pick"):
        extra = [m for m in listed if all(k in m.lower() for k in p["pick"])
                 and not any(a in m.lower() for a in p["avoid"]) and m not in cands]
        extra.sort(reverse=True)  # 版本号大的通常更新
        cands += extra
    if not cands:
        cands = list(p["models"])
    tried = []
    for m in cands[:5]:
        cfg["model"] = m
        try:
            test(cfg, retry_429=False)
            return m, tried
        except LLMError as e:
            msg = str(e)
            tried.append(f"{m}：{msg[:80]}")
            if "401" in msg or "无效" in msg:
                raise LLMError("这个 Key 无效，请检查有没有复制完整（前后不要有空格）")
            if "429" in msg:
                raise LLMError("Key 是有效的，但现在请求太频繁或免费额度用完了（429）。过一会儿再点一键配置试试。")
    raise LLMError("没有找到能用的免费模型：" + "；".join(tried[:3]))
