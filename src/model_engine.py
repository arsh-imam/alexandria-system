import re
"""
ALEXANDRIA Model Engine v2.10 — writing-path guidance + constraint
verifier.

v2.6 over v2.5:
  - Direct writing path now receives explicit task instructions
    (execute exactly, preserve key info, obey limits, output only
    the result).
  - Constraint verifier: explicit, checkable constraints in writing
    requests (character/word limits, sentence counts, item counts)
    are detected, the output is MEASURED against them, and a single
    corrective retry fires on violation. For these tasks generation
    is buffered (not live-streamed) so the retry is invisible; the
    final text is then streamed to the UI.
"""
from llama_cpp import Llama
from retriever import Retriever, RETRIEVAL_CONFIG
from router import classify_query
from compute import detect_and_compute
from deliberate import run_deliberate
import instrumentation as instr
import json, os, re, time
from datetime import datetime

CONFIG = {
    "use_router":           True,
    "use_followup_rewrite": True,
    "use_constraint_check": True,
    "use_compute_tier":     True,
    "use_persona_tier":     True,
    "use_deliberate":       True,
    "active_model":         "qwen3-1.7b",   # "1.5b" | "3b" | "qwen3-1.7b"
    "grounded_logit":       2.0,
    "tentative_logit":      0.5,
    "epistemic_gate_floor": -1.0,   # A Part3: matched #1 specialist may ground down to here
    "src_show_gap":     4.0,   # W3: hide shown source >this logit below top
    "src_show_floor":   0.5,   # W3: hide shown source below this logit
    "temp_grounded":        0.2,
    "temp_default":         0.4,
}

MODELS = {
    "3b":         "/media/pi/KINGSTON/local_ai/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "1.5b":       "/media/pi/KINGSTON/local_ai/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "qwen3-1.7b": "/media/pi/KINGSTON/local_ai/models/Qwen3-1.7B-Q4_K_M.gguf",
}
SESSIONS_DIR = "/media/pi/KINGSTON/local_ai/sessions"

SYSTEM_PROMPT = """You are ALEXANDRIA, a knowledgeable and friendly AI assistant and tutor. Explain clearly and adapt depth to the question. Be concise — short answers to simple questions, thorough answers only when asked for detail.
RULES:
1. Write as if speaking directly to the reader, who sees ONLY your answer. Never mention reference notes, context, sources, or "information provided" — just give the answer naturally.
2. NEVER invent or guess names, dates, numbers, episode counts, rankings, citations, or statistics. If unsure of a specific, say "I'm not certain about the specifics" instead of guessing. A shorter accurate answer beats a longer one with invented details.
3. NEVER write a "Sources:" line yourself — it is added automatically.
4. Answer the EXACT topic asked — not a related but different one."""


# ── Persona tier: identity / social / emotional (no retrieval) ──────
PERSONA_IDENTITY = (
    "You are ALEXANDRIA, a friendly AI assistant. Answer in 1-3 short, warm "
    "sentences, directly. If asked who or what you are, say you are ALEXANDRIA, "
    "an AI assistant that can answer questions, explain topics, help with "
    "writing, and look things up. If asked what you can do, say that briefly. "
    "NEVER say you were made by, created by, trained by, or affiliated with any "
    "company or organization, and never name one (do NOT mention OpenAI, Google, "
    "Anthropic, Alibaba, Meta, Microsoft, or any other). You are NOT ChatGPT, "
    "GPT, Claude, Gemini, Bard, Qwen, or Llama — you are ALEXANDRIA. ONLY if the "
    "user explicitly asks which underlying model or language model powers you, "
    "you may say the underlying model is Qwen3-1.7B; otherwise never mention a "
    "model. Do not list features or mention being offline unless asked. Just "
    "answer naturally."
)

PERSONA_WARM = (
    "You are ALEXANDRIA, a warm, friendly, emotionally intelligent AI assistant "
    "talking with someone casually. Respond like a kind person would, in 1-3 "
    "short sentences. If they share a feeling, acknowledge it genuinely and "
    "gently and invite them to say more if it fits. If they greet you, thank "
    "you, or compliment you, respond warmly and naturally. Do NOT give "
    "definitions, encyclopedic facts, lists, steps, or sources. Do NOT mention "
    "being a model or any company. Just be human, kind, and brief."
)

_IDENTITY_RE = re.compile(
    r"^\s*(who\s+are\s+you|what\s+are\s+you|who're\s+you"
    r"|what'?s?\s+your\s+name|what\s+is\s+your\s+name"
    r"|do\s+you\s+have\s+a\s+name|your\s+name"
    r"|introduce\s+yourself|tell\s+me\s+about\s+yourself"
    r"|who\s+(made|created|built|developed|trained|designed)\s+you"
    r"|are\s+you\s+(an?\s+)?(ai|robot|bot|human|real|conscious|sentient|alive|alexandria)"
    r"|are\s+you\s+(chat\s?gpt|gpt-?\d?|claude|gemini|bard|qwen|llama|copilot|deepseek|grok|alexa|siri)"
    r"|what\s+can\s+you\s+do|what\s+do\s+you\s+do)\s*\??\s*$",
    re.I)

_MODEL_Q_RE = re.compile(
    r"\b(what|which)\s+(model|llm|language\s+model|architecture)\b"
    r"|\bwhat\s+are\s+you\s+(built|based|made|trained|running)\s+on\b"
    r"|\bare\s+you\s+(qwen|gpt|chat\s?gpt|claude|gemini|llama)\b", re.I)

_SOCIAL_RE = re.compile(
    r"^\s*("
    r"hi+|hey+|hello+|yo+|sup|howdy|greetings|hiya|heya"
    r"|good\s+(morning|afternoon|evening|night)|goodnight|good\s+day"
    r"|how\s+(are|r)\s+(you|u)|how'?s\s+it\s+going|how\s+do\s+you\s+do"
    r"|how\s+have\s+you\s+been|what'?s\s+up|whats\s+up|wassup|wazzup"
    r"|nice\s+to\s+meet\s+you|pleasure\s+to\s+meet\s+you"
    r"|thank\s*(you|s)|thanks|thx|ty|tysm|much\s+appreciated|appreciate\s+it"
    r"|good\s+(job|bot|work)|well\s+done|nice\s+(one|work|job)"
    r"|you'?re\s+(awesome|amazing|great|the\s+best|cool|smart|brilliant|helpful|wonderful|nice|stupid|dumb|useless|bad|wrong|annoying)"
    r"|i\s+(love|like|hate)\s+(you|u)|love\s+(you|u)|luv\s+u"
    r"|bye+|goodbye|see\s+(you|ya)|cya"
    r")\s*[\s!.,?]*(you\??)?\s*$",
    re.I)

_EMOTION_RE = re.compile(
    r"\bi'?m\s+(so\s+|really\s+|very\s+|feeling\s+)?"
    r"(sad|depressed|down|unhappy|miserable|lonely|alone|anxious|stressed"
    r"|worried|scared|afraid|terrified|tired|exhausted|angry|upset|frustrated"
    r"|nervous|overwhelmed|heartbroken|hopeless|lost|happy|excited|thrilled"
    r"|great|good|fine|bored|sleepy|grateful|blessed)\b"
    r"|\bi\s+feel\s+(so\s+|really\s+|very\s+)?(sad|down|low|depressed|anxious"
    r"|lonely|stressed|terrible|awful|horrible|great|happy|better|lost|empty"
    r"|hopeless|worthless|overwhelmed|amazing)\b"
    r"|\bi\s+(had|am\s+having)\s+a\s+(bad|rough|terrible|hard|great|good|wonderful)\s+day"
    r"|\bi'?m\s+not\s+(okay|ok|feeling\s+well|doing\s+well|good)\b"
    r"|\bi\s+(can'?t|cannot)\s+sleep\b",
    re.I)

_INFO_SIGNAL = re.compile(
    r"\b(what|what'?s|how|why|when|where|which|who|whom|should|shall"
    r"|can\s+you|could\s+you|would\s+you|recommend|suggest|explain|define"
    r"|tell\s+me\s+(about|how|why|what|the)|help\s+me|list|meaning|difference"
    r"|compare|vs|best\s+way|give\s+me)\b|\?",
    re.I)

WRITING_INSTRUCTION = (
    "This is a writing/editing task. Execute the requested "
    "transformation EXACTLY as asked. Preserve all key information "
    "from the original (deadlines, names, numbers, requirements) "
    "unless removal is requested. If a length or count limit is "
    "given, obey it strictly. Output ONLY the requested text — no "
    "explanations, notes, or preamble unless explicitly asked for.")

FOLLOWUP_HINTS = re.compile(
    r"\b(he|she|his|her|its|they|their|them|that|those|this|these|it|the same)\b"
    r"|^(what about|how about|and|also|tell me more|more|why|when|where)\b",
    re.I)

SOURCES_RE = re.compile(r"\n*\s*(Sources:|Sources \(|Note: answered from model)"
                        r".*\Z", re.S)


def _strip_sources(text):
    return SOURCES_RE.sub("", text).rstrip()


# ── Constraint detection / verification ─────────────────────────────

CHAR_LIMIT  = re.compile(r"(?:under|less than|at most|max(?:imum)?(?: of)?|"
                         r"no more than|within)\s+(\d+)\s+characters?", re.I)
WORD_LIMIT  = re.compile(r"(?:under|less than|at most|max(?:imum)?(?: of)?|"
                         r"no more than|in|within)\s+(\d+)\s+words?", re.I)
ONE_SENT    = re.compile(r"\b(?:in |to |as )?(?:one|a single|1) sentence\b", re.I)
ITEM_COUNT  = re.compile(r"\b(\d+)\s+(?:\w+\s){0,2}?(options?|ideas?|names?|"
                         r"examples?|suggestions?|lines?|items?|titles?|"
                         r"versions?|variations?|bullets?|points?)\b", re.I)


def detect_constraints(query):
    cons = []
    m = CHAR_LIMIT.search(query)
    if m:
        cons.append(("max_chars", int(m.group(1))))
    m = WORD_LIMIT.search(query)
    if m:
        cons.append(("max_words", int(m.group(1))))
    if ONE_SENT.search(query):
        cons.append(("max_sentences", 1))
    m = ITEM_COUNT.search(query)
    if m:
        cons.append(("item_count", int(m.group(1))))
    return cons


def check_constraints(text, cons):
    """Return list of human-readable violations (empty = pass)."""
    body = text.strip()
    fails = []
    for kind, n in cons:
        if kind == "max_chars" and len(body) > n:
            fails.append(f"output is {len(body)} characters but the "
                         f"limit is {n} characters")
        elif kind == "max_words" and len(body.split()) > n:
            fails.append(f"output is {len(body.split())} words but the "
                         f"limit is {n} words")
        elif kind == "max_sentences":
            sents = [s for s in re.split(r"[.!?]+", body) if s.strip()]
            if len(sents) > n:
                fails.append(f"output has {len(sents)} sentences but "
                             f"must be {n} sentence(s)")
        elif kind == "item_count":
            items = re.findall(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+", body)
            if items and len(items) != n:
                fails.append(f"output has {len(items)} items but "
                             f"exactly {n} were requested")
    return fails


class ThinkFilter:
    """Streaming filter that removes <think>...</think> blocks even
    when the tags arrive split across tokens."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def feed(self, piece):
        self.buf += piece
        out = ""
        while True:
            if self.in_think:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    self.buf = self.buf[-(len(self.CLOSE) - 1):]
                    return out
                self.buf = self.buf[i + len(self.CLOSE):]
                self.in_think = False
            else:
                i = self.buf.find(self.OPEN)
                if i == -1:
                    keep = len(self.OPEN) - 1
                    safe = self.buf[:-keep] if len(self.buf) > keep else ""
                    tail = self.buf[len(safe):]
                    if "<" in tail:
                        out += safe
                        self.buf = tail
                    else:
                        out += self.buf
                        self.buf = ""
                    return out
                out += self.buf[:i]
                self.buf = self.buf[i + len(self.OPEN):]
                self.in_think = True

    def flush(self):
        out = "" if self.in_think else self.buf
        self.buf = ""
        return out



def _pretty_title(raw):
    """Clean an article title for display (UI polish; not used for retrieval)."""
    import re as _re
    t = (raw or "").strip()
    # Stack-Exchange scraped URL fragments e.g. "questions/tagged/x_page=4"
    if t.startswith("questions/tagged/"):
        frag = t.split("/", 2)[-1]
        frag = _re.sub(r"_page=\d+$", "", frag)
        frag = frag.replace("-", " ").replace("_", " ").strip()
        return "Discussions on " + frag.title() if frag else "Community discussion"
    # generic leftover url-ish fragments
    if "/" in t and " " not in t and "_page=" in t:
        frag = _re.sub(r"_page=\d+$", "", t.split("/")[-1]).replace("-", " ")
        return frag.title()
    # namespace prefixes from wikibooks/wikiversity
    for pref in ("Cookbook:", "Wikijunior:", "Subject:", "Portal:"):
        if t.startswith(pref):
            t = t[len(pref):].strip()
    # "A/B/C" structured wikibook titles -> last meaningful segment
    if "/" in t and len(t) > 0 and not t.lower().startswith("http"):
        segs = [seg for seg in t.split("/") if seg.strip()]
        if len(segs) > 1:
            t = segs[-1].strip()
    return t if t else "source"

def _pretty_source(raw):
    """Display name for a corpus label (UI polish; not used for retrieval)."""
    s = (raw or "").strip()
    low = s.lower()
    # strip internal plumbing
    low = low.replace("(deep)", "").replace("deep archive", "").strip()
    low = re.sub(r"\s+", " ", low)
    EXACT = {
        "wikipedia": "Wikipedia",
        "wikipedia medicine": "Wikipedia (Medicine)",
        "wikibooks": "Wikibooks", "wikiquote": "Wikiquote",
        "wikivoyage": "Wikivoyage", "wiktionary": "Wiktionary",
        "wikinews": "Wikinews", "wikiversity": "Wikiversity",
        "ifixit": "iFixit",
    }
    if low in EXACT:
        return EXACT[low]
    # "<topic> se (community q&a)" -> "<Topic> Stack Exchange"
    m = re.match(r"(.+?)\s*se\s*\(community\s*q&a\)", low)
    if m:
        topic = m.group(1).strip()
        SPECIAL = {"diy": "DIY", "scifi": "Sci-Fi", "askubuntu": "Ask Ubuntu",
                   "superuser": "Super User", "serverfault": "Server Fault",
                   "ux": "UX", "rpg": "RPG", "gis": "GIS", "ell": "ELL"}
        topic_disp = SPECIAL.get(topic, topic.title())
        return topic_disp + " Stack Exchange"
    if low.endswith("se") and len(low) > 2:
        return low[:-2].strip().title() + " Stack Exchange"
    return s if s else "source"


class ModelEngine:

    def __init__(self):
        self.llm       = None
        self.retriever = Retriever()
        self.conversation_history = []
        self.session_id = None
        self.anchor_titles = []
        self.is_qwen3 = CONFIG["active_model"].startswith("qwen3")

    def load(self):
        print(f"Loading language model ({CONFIG['active_model']})...",
              flush=True)
        self.llm = Llama(
            model_path=MODELS[CONFIG["active_model"]], n_ctx=4096,
            n_threads=4, n_batch=512, verbose=False)
        print("Model loaded.", flush=True)
        self.retriever.load()
        self.new_conversation()

    def new_conversation(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        sp = SYSTEM_PROMPT + (" /no_think" if self.is_qwen3 else "")
        self.conversation_history = [{"role": "system", "content": sp}]
        self.anchor_titles = []

    # ── Helpers ────────────────────────────────────────────────────

    def _count_tokens(self, text):
        try:
            return len(self.llm.tokenize(text.encode("utf-8")))
        except Exception:
            return int(len(text.split()) * 1.3)

    def _max_tokens_for(self, query, depth, skip_retrieval):
        if depth == "detailed":
            return 700
        if depth == "brief":
            return 200
        if skip_retrieval:
            return 550
        if len(query.split()) > 40:
            return 550
        return 350

    def _looks_like_followup(self, query):
        if len(self.conversation_history) < 3:
            return False
        if len(query.split()) > 14:
            return False
        return bool(FOLLOWUP_HINTS.search(query))

    def _rewrite_followup(self, query):
        recent = []
        for m in self.conversation_history[1:][-4:]:
            recent.append(f"{m['role']}: {_strip_sources(m['content'])[:300]}")
        prompt = (
            "Conversation so far:\n" + "\n".join(recent) +
            f"\n\nThe user now asks: \"{query}\"\n"
            "Rewrite this as ONE standalone search query that includes "
            "the specific names/topics it refers to. Use ONLY names, "
            "numbers, and facts that appear in the conversation above "
            "— NEVER add new ones. Output ONLY the rewritten query, "
            "nothing else."
            + (" /no_think" if self.is_qwen3 else ""))
        try:
            out = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80, temperature=0.1, stream=False)
            text = out["choices"][0]["message"]["content"]
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            text = text.strip().strip('"').strip()
            if 0 < len(text.split()) <= 25:
                return text
        except Exception:
            pass
        return query

    def _decompose_query(self, query):
        prompt = (
            f"Question: \"{query}\"\n"
            "Break this into AT MOST 2 simple, standalone search "
            "queries that together find the facts needed. Use ONLY "
            "words and concepts from the question — do NOT add names "
            "or facts of your own. Output one query per line, "
            "nothing else."
            + (" /no_think" if self.is_qwen3 else ""))
        try:
            out = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80, temperature=0.1, stream=False)
            text = out["choices"][0]["message"]["content"]
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            subs = [l.strip().strip('"').lstrip('0123456789.-) ')
                    for l in text.splitlines() if l.strip()]
            subs = [s for s in subs if 2 <= len(s.split()) <= 20][:2]
            return subs
        except Exception:
            return []

    def _deliberate_retrieve(self, query, status_callback):
        """Decompose -> retrieve per sub-question -> fuse evidence."""
        if status_callback:
            status_callback("Breaking question into parts...", "#d4900a")
        subs = self._decompose_query(query)
        all_results = []
        seen = set()
        for i, sq in enumerate(subs, 1):
            if status_callback:
                status_callback(
                    f"Researching part {i}/{len(subs)}: {sq[:40]}...",
                    "#d4900a")
            res, _, _ = self.retriever.search(sq)
            for r in res:
                key = (r["title"], r["text"][:80])
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)
        # also keep the original query's own retrieval
        if status_callback:
            status_callback("Combining the evidence...", "#d4900a")
        res, conf, st = self.retriever.search(query)
        for r in res:
            key = (r["title"], r["text"][:80])
            if key not in seen:
                seen.add(key)
                all_results.append(r)
        all_results.sort(key=lambda r: -r.get("logit", -20))
        return all_results[:5], subs

    def _persona_route(self, query):
        """Return 'identity' | 'social' | 'emotional' | None (no model call)."""
        q = query.strip()
        if not q or len(q) > 90:
            return None
        if _IDENTITY_RE.match(q):
            return "identity"
        if _MODEL_Q_RE.search(q) and len(q.split()) <= 10:
            return "identity"
        if _SOCIAL_RE.match(q):
            return "social"
        if (_EMOTION_RE.search(q) and len(q.split()) <= 9
                and not _INFO_SIGNAL.search(q)):
            return "emotional"
        return None

    def _generate_text(self, messages, max_tokens, temperature):
        """Non-streamed generation, think-tags stripped."""
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, top_p=0.9, repeat_penalty=1.1,
            stream=False)
        text = out["choices"][0]["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return text.strip()

    # ── Main generation ────────────────────────────────────────────

    def generate_stream(self, user_message, token_callback,
                        status_callback=None):
        rec = instr.new_record(user_message, path="unknown")
        rec["model"] = CONFIG["active_model"]
        t_start = time.time()

        if status_callback:
            status_callback("Analyzing query...", "#1d9e75")
        route = classify_query(user_message) if CONFIG["use_router"] else {
            "query_type": "general", "skip_retrieval": False,
            "depth": "simple", "source_weights": {}}
        rec["query_type"]     = route["query_type"]
        rec["skip_retrieval"] = route["skip_retrieval"]
        max_tokens = self._max_tokens_for(
            user_message, route["depth"], route["skip_retrieval"])

        # ── Compute tier: exact answers for computable queries ──
        computed = (detect_and_compute(user_message)
                    if CONFIG["use_compute_tier"] else None)
        if computed:
            rec["path"] = "compute"
            rec["zone"] = "computed"
            rec["computed"] = computed
            if status_callback:
                status_callback("Computing exact answer...", "#1d9e75")
            messages = [self.conversation_history[0]]
            messages.append({"role": "system", "content":
                "VERIFIED COMPUTATION (exact, calculated by the "
                "system's calculator): " + computed + ". State this "
                "result clearly and naturally. Do NOT recalculate or "
                "alter the numbers."})
            messages.append({"role": "user", "content": user_message})
            t_gen0 = time.time()
            text = self._generate_text(messages, 150, 0.1)
            first_token_time = time.time()
            token_callback(text)
            note = "\n\nComputed exactly by ALEXANDRIA calculator."
            token_callback(note)
            full_reply = text + note
            t_end = time.time()
            rec["output_tokens"] = self._count_tokens(text)
            rec["time_to_first_token"] = round(first_token_time - t_start, 2)
            rec["decode_time"] = round(t_end - t_gen0, 2)
            rec["total_time"] = round(t_end - t_start, 2)
            rec["tokens_per_second"] = 0.0
            instr.log_query(rec)
            self.conversation_history.append(
                {"role": "user", "content": user_message})
            self.conversation_history.append(
                {"role": "assistant", "content": full_reply})
            self._save_session()
            return full_reply

        # ── Persona tier: identity / social / emotional (no retrieval) ──
        persona = (self._persona_route(user_message)
                   if CONFIG.get("use_persona_tier", True) else None)
        if persona:
            rec["path"] = "persona"
            rec["zone"] = persona
            if status_callback:
                status_callback("Responding...", "#1d9e75")
            sp = PERSONA_IDENTITY if persona == "identity" else PERSONA_WARM
            temp = 0.2 if persona == "identity" else 0.6
            if self.is_qwen3:
                sp = sp + " /no_think"
            messages = [{"role": "system", "content": sp}]
            for m in self.conversation_history[1:][-4:]:
                messages.append({"role": m["role"],
                                 "content": _strip_sources(m["content"])})
            messages.append({"role": "user", "content": user_message})
            t_gen0 = time.time()
            text = self._generate_text(messages, 160, temp)
            first_token_time = time.time()
            token_callback(text)
            full_reply = text
            t_end = time.time()
            rec["output_tokens"] = self._count_tokens(text)
            rec["time_to_first_token"] = round(first_token_time - t_start, 2)
            rec["decode_time"] = round(t_end - t_gen0, 2)
            rec["total_time"] = round(t_end - t_start, 2)
            rec["tokens_per_second"] = 0.0
            instr.log_query(rec)
            self.conversation_history.append(
                {"role": "user", "content": user_message})
            self.conversation_history.append(
                {"role": "assistant", "content": full_reply})
            self._save_session()
            return full_reply

        search_query = user_message
        is_followup  = False
        if (CONFIG["use_followup_rewrite"]
                and not route["skip_retrieval"]
                and self._looks_like_followup(user_message)):
            if status_callback:
                status_callback("Understanding follow-up...", "#1d9e75")
            search_query = self._rewrite_followup(user_message)
            is_followup  = True
        rec["search_query"] = search_query

        context_block  = ""
        source_summary = ""
        zone           = "direct"
        temperature    = CONFIG["temp_default"]

        if not route["skip_retrieval"] and self.retriever.ready:
            if status_callback:
                status_callback("Searching knowledge sources...", "#1d9e75")
            t_search0 = time.time()
            deliberate = (CONFIG["use_deliberate"]
                          and route.get("compositional"))
            if deliberate:
                if status_callback:
                    status_callback(
                        "Thinking it through (this may take a "
                        "minute)...", "#d4900a")
                results, subs, dtrace = run_deliberate(
                    self.llm, self.retriever, search_query,
                    self.is_qwen3, status_cb=status_callback)
                rec["deliberate"] = True
                rec["sub_queries"] = subs
                rec["deliberate_trace"] = dtrace
                rec["deliberate_forced"] = True
                confidence = (max((r["score"] for r in results),
                                  default=0.0))
                search_time = round(time.time() - t_search0, 3)
                self.retriever.last_diag["top_logit"] = (
                    max((r["logit"] for r in results), default=-20.0))
            else:
                results, confidence, search_time = self.retriever.search(
                    search_query,
                    anchor_titles=(self.anchor_titles if is_followup
                                   else None))
            diag = self.retriever.last_diag
            top_logit  = diag.get("top_logit", -20.0)
            separation = diag.get("separation", 0.0)

            rec["rag_used"]    = bool(results)
            rec["confidence"]  = confidence
            rec["top_logit"]   = top_logit
            rec["separation"]  = separation
            rec["search_time"] = search_time
            rec["num_results"] = len(results)
            rec["sources"]     = list({r["source"] for r in results})
            rec["tiers_used"]  = list({r["tier"] for r in results})

            epi_on = RETRIEVAL_CONFIG.get("use_epistemic_fusion")
            epi_in_top = diag.get("epi_specialist_in_top", False)
            best_spec = diag.get("epi_best_specialist_logit", -20.0)
            best_enc  = diag.get("epi_best_encyc_logit", -20.0)
            if results and top_logit >= CONFIG["grounded_logit"]:
                zone = "grounded"
            elif results and top_logit >= CONFIG["tentative_logit"]:
                zone = "tentative"
            elif (epi_on and epi_in_top and results
                  and best_spec >= CONFIG["epistemic_gate_floor"]
                  and best_enc < CONFIG["grounded_logit"]):
                # Conservative promotion: a domain-matched specialist
                # (e.g. a fact-checking source for a myth query) is in
                # the top results above the epistemic floor, AND no
                # encyclopedic source already grounds the answer. Only
                # ever RAISES a would-be-unverified answer; never
                # demotes, never overrides a strong Wikipedia answer.
                zone = "grounded"
                rec["epistemic_promotion"] = True
            else:
                zone = "unverified"
            rec["zone"] = zone
            rec["grounded"] = zone == "grounded"
            if zone == "grounded":
                self.anchor_titles = list(dict.fromkeys(
                    [r["title"] for r in results]))[:3]

            followup_nudge = (
                " The user is asking a FOLLOW-UP: add NEW information "
                "beyond what you already said; do not repeat your "
                "previous answer." if is_followup else "")

            if zone in ("grounded", "tentative"):
                temperature = CONFIG["temp_grounded"]
                if any(r["tier"] == 2 for r in results) and status_callback:
                    status_callback("Deep archive search...", "#d4900a")
                lines = []
                for r in results:
                    words = r["text"].split()[:120]
                    lines.append(f"[{r['title']}] ({r['source']})\n"
                                 + ' '.join(words))
                if zone == "grounded":
                    instruction = (
                        "Answer the question using the reference notes "
                        "below as your factual basis, but write directly "
                        "to the reader without mentioning the notes. If a "
                        "note is about a DIFFERENT place, person, or thing "
                        "than the question asks, ignore it. If you are not "
                        "sure of a specific, say so rather than guessing.")
                else:
                    instruction = (
                        "The reference notes below may only partly fit the "
                        "question. Use any parts that clearly apply, ignore "
                        "the rest, and do NOT invent specific names, places, "
                        "or figures beyond what you are sure of. Write "
                        "directly to the reader without mentioning the notes. "
                        "It is fine to keep the answer general.")
                context_block = (
                    "Reference notes:\n\n" + "\n\n".join(lines) + "\n\n"
                    + instruction
                    + " Do NOT write a Sources line — it is added "
                      "automatically." + followup_nudge)
                # W3: only DISPLAY sources that earned their place — drop
                # any whose rerank logit is far below the top or below a
                # floor (display-only; does not affect context or ranking).
                _top = results[0]["logit"] if results else 0.0
                _gap = CONFIG.get("src_show_gap", 4.0)
                _floor = CONFIG.get("src_show_floor", 0.5)
                seen, src_parts = set(), []
                for _i, r in enumerate(results):
                    key = (r["title"], r["source"])
                    if key in seen:
                        continue
                    # W3b: never DISPLAY a disambiguation page — it is a nav
                    # stub, never the article that answered (display-only).
                    _t = r["title"].lower()
                    if _i > 0 and _t.endswith("(disambiguation)"):
                        continue
                    # always keep the #1 source; gate the rest
                    if _i > 0 and (_top - r["logit"] > _gap
                                   or r["logit"] < _floor):
                        continue
                    seen.add(key)
                    src_parts.append(f"{_pretty_title(r['title'])} \u2014 {_pretty_source(r['source'])}")
                label = ("Sources: " if zone == "grounded"
                         else "Sources (partial match): ")
                source_summary = label + " · ".join(src_parts[:4])
            else:
                context_block = (
                    "No reliable offline sources were found for this "
                    "question. Answer from your own general knowledge. "
                    "Be appropriately careful with specifics, and do "
                    "not invent precise figures or names. Do NOT write "
                    "a Sources line." + followup_nudge)
                source_summary = ("Note: answered from model knowledge — "
                                  "no strong offline source found.")
            rec["path"] = "rag"
        else:
            rec["path"] = "direct"
            rec["zone"] = "direct"
            if route["query_type"] == "writing":
                context_block = WRITING_INSTRUCTION

        messages = [self.conversation_history[0]]
        for m in self.conversation_history[1:][-6:]:
            messages.append({"role": m["role"],
                             "content": _strip_sources(m["content"])})
        if context_block:
            messages.append({"role": "system", "content": context_block})
        messages.append({"role": "user", "content": user_message})
        rec["input_tokens"] = sum(
            self._count_tokens(m["content"]) for m in messages)
        rec["temperature"] = temperature

        # ── Constraint-verified path (writing tasks with checkable
        #    limits): buffered generate -> verify -> retry once ──────
        constraints = (detect_constraints(user_message)
                       if (CONFIG["use_constraint_check"]
                           and route["query_type"] == "writing")
                       else [])
        rec["constraints"] = [f"{k}={n}" for k, n in constraints]

        if constraints:
            if status_callback:
                status_callback("Writing (checking limits)...", "#1d9e75")
            t_gen0 = time.time()
            draft = self._generate_text(messages, max_tokens, temperature)
            fails = check_constraints(draft, constraints)
            rec["constraint_retry"] = bool(fails)
            if fails:
                if status_callback:
                    status_callback("Fixing length/count...", "#d4900a")
                fix_messages = messages + [
                    {"role": "assistant", "content": draft},
                    {"role": "user", "content":
                        "Your answer broke the stated limit: "
                        + "; ".join(fails) +
                        ". Rewrite it to satisfy the limit exactly. "
                        "Output ONLY the corrected text."
                        + (" /no_think" if self.is_qwen3 else "")}]
                retry = self._generate_text(
                    fix_messages, max_tokens, 0.2)
                if not check_constraints(retry, constraints):
                    draft = retry
                elif len(check_constraints(retry, constraints)) < len(fails):
                    draft = retry
            rec["constraint_ok"] = not check_constraints(draft, constraints)
            first_token_time = time.time()
            token_callback(draft)
            full_reply = draft
            out_tokens = self._count_tokens(draft)
            t_end = time.time()
            ttft   = first_token_time - t_start
            decode = t_end - t_gen0
        else:
            if status_callback:
                status_callback("Generating response...", "#1d9e75")
            stream = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, top_p=0.9, repeat_penalty=1.1,
                stream=True)
            tf = ThinkFilter() if self.is_qwen3 else None
            full_reply       = ""
            first_token_time = None
            out_tokens       = 0
            for chunk in stream:
                piece = chunk["choices"][0]["delta"].get("content")
                if piece:
                    out_tokens += 1
                    if tf:
                        piece = tf.feed(piece)
                    if piece:
                        if first_token_time is None:
                            first_token_time = time.time()
                        full_reply += piece
                        token_callback(piece)
            if tf:
                rest = tf.flush()
                if rest:
                    full_reply += rest
                    token_callback(rest)
            t_end = time.time()
            ttft   = (first_token_time - t_start) if first_token_time else 0.0
            decode = (t_end - first_token_time) if first_token_time else 0.0

        if source_summary:
            token_callback("\n\n" + source_summary)
            full_reply += "\n\n" + source_summary

        rec["output_tokens"]       = out_tokens
        rec["time_to_first_token"] = round(ttft, 2)
        rec["decode_time"]         = round(decode, 2)
        rec["total_time"]          = round(t_end - t_start, 2)
        rec["tokens_per_second"]   = (
            round(out_tokens / decode, 2) if decode > 0 else 0.0)
        instr.log_query(rec)

        self.conversation_history.append(
            {"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": full_reply})
        self._save_session()
        return full_reply

    # ── Sessions ────────────────────────────────────────────────────

    def _save_session(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        path = os.path.join(SESSIONS_DIR, f"session_{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"session_id": self.session_id,
                       "history": self.conversation_history},
                      f, ensure_ascii=False, indent=2)

    def list_sessions(self):
        if not os.path.exists(SESSIONS_DIR):
            return []
        out = []
        for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(SESSIONS_DIR, fname)) as f:
                    data = json.load(f)
                title = next(
                    (m["content"][:55] for m in data["history"]
                     if m["role"] == "user"), "Empty session")
                out.append({"id": data["session_id"], "title": title})
            except Exception:
                pass
        return out

    def load_session(self, session_id):
        path = os.path.join(SESSIONS_DIR, f"session_{session_id}.json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        self.session_id           = data["session_id"]
        self.conversation_history = data["history"]
        return [m for m in data["history"] if m["role"] != "system"]
