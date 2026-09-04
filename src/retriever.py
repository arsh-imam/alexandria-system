"""
LCATRS Retrieval Core v3 — federated deep archive with semantic
routing.

Generators (flag-toggleable):
  1. Title match   — FTS5 on title column
  2. BM25 / FTS5   — lexical full text
  3. Dense vectors — Annoy + aligned embedder
  4. FEDERATED deep archives — Xapian over: Wikipedia (ALWAYS) +
     top-k specialty archives chosen by query-to-description
     similarity (floor 0.45). Each candidate carries its archive's
     provenance label. All candidates face the same cross-encoder
     arbitration — new sources can only win by outscoring on the
     same question (the non-regression guarantee).
  5. Anchors — previous grounded turn's articles, on follow-ups
"""
import os
import re
import time
import math
import sqlite3
import numpy as np
from annoy import AnnoyIndex

from embedder import AlignedEmbedder
from reranker import Reranker
from archive_registry import ArchiveRegistry

ANNOY_FILE = "/media/pi/KINGSTON/local_ai/index/passages.ann"
DB_FILE    = "/media/pi/KINGSTON/local_ai/index/passages.db"
DIMS       = 384

RETRIEVAL_CONFIG = {
    "use_title":         True,
    "use_bm25":          True,
    "use_vector":        True,
    "use_anchors":       True,
    "deep_archive":      "always",   # "always" | "off"
    "route_k":           3,
    "route_floor":       0.50,
    "title_k":           8,
    "bm25_k":            20,
    "vector_k":          15,
    "deep_articles":     6,          # W2a: union cap, Wikipedia (always)
    "deep_articles_spec": 3,         # W2a: union cap per routed specialty
    "deep_chunks":       2,
    "deep_chunks_wiki":  6,     # 3b: windows scanned per Wikipedia article
    "deep_locate":       True,  # 3b: locate matched passage, not just lead
    "deep_per_query":    3,     # W2a: ZIM hits taken per sub-query
    "deep_multiquery":   True,  # W2a: union raw+rare+entity sub-queries
    "entity_titles":     True,  # P1: fetch the named entity's article
    "entity_spans_ci":   True,  # case-insensitive title-validated entity spans
    "entity_span_max":   3,     # max entity spans returned
    "entity_span_checks": 10,   # max title-validation probes per query
    "anchor_chunks":     2,
    "final_k":           3,
    "max_per_title":     2,
    "grounded_logit":    2.0,
    "ablation_sources": "all",       # "all" | "wiki_deep" | "wiki_pure"
    "use_epistemic_fusion": False,   # A: source-epistemology fusion (OFF=baseline)
    "epistemic_affinity_bonus": 1.0, # additive only; moves near-ties
    "medical_affinity":  True,  # W4a: prefer relevant Wikipedia-medical on health Qs
    "medical_bonus":     4.0,
    "medical_floor":     0.0,
    "authority_bonus":   0.5,
    "strip_boilerplate":  True,    # B: drop nav/sidebar/infobox cruft
    "boiler_min_words":   18,
    "boiler_func_ratio":  0.16,
    "boiler_lower_ratio": 0.22,
}

STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','being','what',
    'who','why','how','when','where','which','does','do','did','of',
    'in','on','at','to','and','or','it','its','between','mean','means',
    'his','her','their','my','your','i','you','for','with','as','by',
    'this','that','these','those','about','can','could','should',
    'would','will','please','tell','me','explain','describe','give',
    'know','some','any','from','into','than','then','there','here',
    'we','us','our','ours','have','has','had','having','get','gets',
    'got','getting','many','much',
}


def _content_terms(query):
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 2]


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_STRIPCHARS = '.,;:!?()[]' + chr(34) + chr(39)


def _cap_spans(text):
    """Capitalized multi-word spans from the raw query (named-entity signal)."""
    spans = re.findall(
        r"\b[A-Z][A-Za-z0-9'.\-]*(?:\s+(?:and|of|the|in|on|&|[A-Z][A-Za-z0-9'.\-]*))+",
        text)
    return [s.strip() for s in spans if len(s.strip()) >= 2]



# ── Epistemic-need classification (for optional fusion, A) ──────────
_NEED_PATTERNS = {
    "claimcheck": [r"\bis it true\b", r"\bis it real\b", r"\breally\b.*\?",
                   r"\bmyth\b", r"\bdebunk", r"\bactually (true|cause|work)",
                   r"\bdoes .+ (really|actually)\b", r"\bcommon misconception\b",
                   r"\bis it (bad|good|safe|dangerous|healthy) (for|to)\b",
                   r"\bdoes .+ (cause|cure|prevent|improve|boost|increase|reduce|help|make)\b",
                   r"\bcan .+ (cause|cure|prevent|kill|damage)\b",
                   r"\b(do|does) .+ (work|help)\b.*\?",
                   r"\bare .+ (bad|good|harmful|safe) for\b",
                   r"\bwill .+ (hurt|harm|damage|help|cause)\b"],
    "medical":    [r"\b(headaches?|migraines?|fever|sore[ -]?throat|coughing?|"
                   r"common cold|flu|influenza|nausea|nauseous|vomit\w*|"
                   r"diarrh\w+|constipat\w+|heartburn|dizz\w+|vertigo|"
                   r"blood pressure|hypertension|insomnia|trouble sleeping|"
                   r"improve my sleep|sleep better|rash|hives|sprains?|burns?|"
                   r"allerg\w+|congestion|sinus\w*|stomach ?ache|toothache|"
                   r"earache|cramps?|bloat\w+|dehydrat\w+|hangover|sunburn|"
                   r"indigestion|nosebleed|heatstroke)\b",
                   r"can'?t sleep"],
    "howto":      [r"\bhow (do|can|to) (i|you|we)\b", r"\bhow do i\b",
                   r"\bfix\b", r"\brepair\b", r"\bsteps to\b",
                   r"\bbest way to\b", r"\bhow should i\b", r"\bget rid of\b"],
}
def _epistemic_need(query):
    import re as _re
    q = query.lower()
    for need, pats in _NEED_PATTERNS.items():
        if any(_re.search(p, q) for p in pats):
            return need
    return "factual"


# Archives force-included for a given epistemic need (bypasses the
# similarity floor so the right specialist is ALWAYS in the pool).
_FORCE_ON_NEED = {
    "claimcheck": ["skeptics"],
}

# need -> source etypes that deserve a relevance bonus
_NEED_AFFINITY = {
    "claimcheck": {"debunking", "encyclopedic"},
    "howto":      {"instructional"},
    "factual":    {"encyclopedic"},
}

# map a source LABEL back to an etype (labels carry the archive identity)
def _label_etype(source_label, registry):
    if registry is None:
        return "encyclopedic"
    for _n, _e in registry.entries.items():
        if _e["label"] == source_label:
            return _e["etype"]
    # fast-index sources (ifixit/wikibooks/etc.) are instructional/encyclopedic
    if source_label in ("ifixit",):
        return "instructional"
    if source_label in ("wikibooks", "wikiversity"):
        return "instructional"
    return "encyclopedic"


class Retriever:

    def __init__(self):
        self.ready    = False
        self.index    = None
        self.db       = None
        self.embedder = None
        self.reranker = None
        self.registry = None
        self._df_cache = {}
        self.last_diag = {}

    def load(self):
        if not os.path.exists(ANNOY_FILE) or not os.path.exists(DB_FILE):
            print("Retriever: index or database missing — RAG disabled.")
            return

        print("Loading Annoy index (mmap)...", flush=True)
        self.index = AnnoyIndex(DIMS, 'angular')
        self.index.load(ANNOY_FILE, prefault=False)
        print(f"  {self.index.get_n_items():,} vectors.", flush=True)

        print("Connecting to passage database...", flush=True)
        self.db = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute(
                "SELECT rowid FROM passages_fts LIMIT 1").fetchone()
            self.has_fts = True
            print("  FTS5 (BM25) table found.", flush=True)
        except Exception:
            self.has_fts = False
            print("  WARNING: no FTS5 table — BM25 disabled.", flush=True)

        print("Loading aligned embedder...", flush=True)
        self.embedder = AlignedEmbedder()
        self.embedder.embed_one("warmup")

        print("Loading cross-encoder reranker...", flush=True)
        self.reranker = Reranker()
        self.reranker.score("warmup", ["warmup passage"])

        print("Opening federated archives...", flush=True)
        try:
            self.registry = ArchiveRegistry().load(self.embedder)
        except Exception as e:
            print(f"  registry failed: {e}", flush=True)
            self.registry = None

        self.ready = True
        print("Retriever v3 ready.", flush=True)

    # ── Generator 1: title match ────────────────────────────────────

    def _entity_spans(self, query):
        """Entity spans for the query, case-INSENSITIVE. Starts with
        capitalized spans (precise), then adds lowercase n-grams that
        actually match an article title (self-validating -> no over-fire).
        Cached per query. Used by title gen and deep multi-query search."""
        if getattr(self, "_espan_q", None) == query:
            return self._espan_v
        cfg = RETRIEVAL_CONFIG
        spans, seen = [], set()
        for s in _cap_spans(query):
            sl = s.lower().strip()
            if sl and sl not in seen:
                seen.add(sl); spans.append(s)
        if cfg.get("entity_spans_ci", True) and self.has_fts:
            toks = re.findall(r"[A-Za-z0-9'&.-]+", query)
            cand = []
            for size in (4, 3, 2):
                for i in range(len(toks) - size + 1):
                    gram = toks[i:i + size]
                    if (gram[0].lower() in STOPWORDS
                            or gram[-1].lower() in STOPWORDS):
                        continue
                    ct = [w for w in gram
                          if w.lower() not in STOPWORDS and len(w) >= 2]
                    if len(ct) >= 2:
                        cand.append((" ".join(gram), ct))
            checks = 0
            maxspan = cfg.get("entity_span_max", 3)
            maxchecks = cfg.get("entity_span_checks", 10)
            for gram, ct in cand:
                if len(spans) >= maxspan or checks >= maxchecks:
                    break
                gl = gram.lower().strip()
                if gl in seen:
                    continue
                match = " AND ".join(f'"{w.lower()}"' for w in ct)
                checks += 1
                try:
                    rows = self.db.execute(
                        "SELECT title FROM passages_fts WHERE title MATCH ? "
                        "ORDER BY bm25(passages_fts, 1.0, 0.0) LIMIT 3",
                        (match,)).fetchall()
                except Exception:
                    rows = []
                # keep ONLY if some matched title is essentially THIS phrase
                # (title content-terms == span terms, allowing <=1 extra) —
                # rejects verb phrases / loose matches that pull long titles.
                want = set(w.lower() for w in ct)
                tight = False
                for r in rows:
                    tct = set(_content_terms(r["title"]))
                    if want <= tct and len(tct - want) <= 1:
                        tight = True
                        break
                if tight:
                    seen.add(gl); spans.append(gram)
        self._espan_q = query; self._espan_v = spans
        return spans

    def _title_candidates(self, query, k):
        if not self.has_fts:
            return []
        terms = _content_terms(query)
        if not terms:
            return []
        cfg = RETRIEVAL_CONFIG
        ids, seen = [], set()
        attempts = []
        if cfg.get("entity_titles", True):
            # entity spans (case-insensitive, title-validated) = strong signal
            for span in self._entity_spans(query):
                st = _content_terms(span)
                if len(st) >= 2:
                    attempts.append(" AND ".join(f'"{t}"' for t in st))
        if len(terms) >= 2:
            attempts.append(" AND ".join(f'"{t}"' for t in terms))
        rare = sorted(terms, key=self._doc_freq)
        if cfg.get("entity_titles", True):
            if len(rare) >= 2:
                attempts.append(f'"{rare[0]}" AND "{rare[1]}"')
            if len(rare) >= 3:
                attempts.append(f'"{rare[0]}" AND "{rare[1]}" AND "{rare[2]}"')
        for t in rare[:2]:
            attempts.append(f'"{t}"')
        for match in attempts:
            try:
                rows = self.db.execute(
                    "SELECT rowid FROM passages_fts WHERE title MATCH ? "
                    "ORDER BY bm25(passages_fts, 1.0, 0.0) LIMIT ?",
                    (match, k)).fetchall()
            except Exception:
                rows = []
            for r in rows:
                if r["rowid"] not in seen:
                    seen.add(r["rowid"]); ids.append(r["rowid"])
            if len(ids) >= k:
                break
        return ids[:k]

    # ── Generator 2: BM25 full text ─────────────────────────────────

    def _doc_freq(self, term):
        if term in self._df_cache:
            return self._df_cache[term]
        try:
            n = self.db.execute(
                "SELECT COUNT(*) FROM passages_fts WHERE passages_fts "
                "MATCH ?", (f'"{term}"',)).fetchone()[0]
        except Exception:
            n = 10 ** 9
        self._df_cache[term] = n
        return n

    def _bm25_candidates(self, query, k):
        if not self.has_fts:
            return []
        terms = _content_terms(query)
        if not terms:
            return []
        if len(terms) > 3:
            terms = sorted(terms, key=self._doc_freq)[:3]
        for joiner in (" AND ", " OR "):
            match = joiner.join(f'"{t}"' for t in terms)
            try:
                rows = self.db.execute(
                    "SELECT rowid FROM passages_fts WHERE passages_fts "
                    "MATCH ? ORDER BY bm25(passages_fts, 10.0, 1.0) "
                    "LIMIT ?", (match, k)).fetchall()
            except Exception:
                rows = []
            if rows:
                return [r["rowid"] for r in rows]
        return []

    # ── Generator 3: dense vectors ──────────────────────────────────

    def _vector_candidates(self, qvec, k):
        return self.index.get_nns_by_vector(qvec, k)

    # ── Generator 4: FEDERATED deep archives ────────────────────────

    def _strip_nav(self, text):
        cfg = RETRIEVAL_CONFIG
        if not cfg.get("strip_boilerplate", True):
            return text
        min_w = cfg.get("boiler_min_words", 18)
        fr_th = cfg.get("boiler_func_ratio", 0.12)
        lr_th = cfg.get("boiler_lower_ratio", 0.35)
        kept = []
        for seg in _SENT_SPLIT.split(text):
            toks = seg.split()
            n = len(toks)
            if n < min_w:
                kept.append(seg)
                continue
            low = sum(1 for t in toks if t.isalpha() and t.islower())
            func = sum(1 for t in toks
                       if t.lower().strip(_STRIPCHARS) in STOPWORDS)
            if (func / n) < fr_th or (low / n) < lr_th:
                continue
            kept.append(seg)
        return ' '.join(kept).strip()

    def _clean_html(self, raw):
        raw = re.sub(r'<style[^>]*>.*?</style>', ' ', raw, flags=re.S)
        raw = re.sub(r'<script[^>]*>.*?</script>', ' ', raw, flags=re.S)
        raw = re.sub(r'<[^>]+>', ' ', raw)
        raw = re.sub(r'&[a-zA-Z#0-9]+;', ' ', raw)
        flat = re.sub(r'\s+', ' ', raw).strip()
        stripped = self._strip_nav(flat)
        return stripped if len(stripped.split()) >= 50 else flat

    def _article_windows(self, words, qterms, n_keep):
        """Pick up to n_keep windows from a deep article: lead +
        keyword-anchored (where query terms appear, rarer-weighted) +
        evenly-spaced (so semantic-only queries still reach deep
        sections). The cross-encoder then arbitrates which win."""
        from collections import Counter
        W, STEP = 160, 120
        windows = []
        for i in range(0, max(1, len(words)), STEP):
            piece = words[i:i + W]
            if len(piece) >= 40:
                windows.append((i, piece))
        if not windows:
            return []
        if len(windows) <= n_keep:
            return [' '.join(p) for _, p in windows]
        cnt = Counter(w.lower().strip(_STRIPCHARS) for w in words)
        def kw_score(piece):
            toks = set(w.lower().strip(_STRIPCHARS) for w in piece)
            s = 0.0
            for t in qterms:
                if t in toks:
                    s += 1.0 / (1.0 + math.log(1 + cnt.get(t, 0)))
            return s
        chosen = {windows[0][0]: windows[0][1]}
        for s, i, p in sorted(((kw_score(p), i, p) for i, p in windows),
                              key=lambda x: -x[0]):
            if s <= 0 or len(chosen) >= n_keep:
                break
            chosen.setdefault(i, p)
        if len(chosen) < n_keep:
            for j in np.linspace(0, len(windows) - 1,
                                 num=min(n_keep, len(windows))):
                i, p = windows[int(round(j))]
                chosen.setdefault(i, p)
                if len(chosen) >= n_keep:
                    break
        return [' '.join(p) for _, p in sorted(chosen.items())]

    def _search_one_archive(self, name, query, n_articles, n_chunks):
        entry = self.registry.get(name) if self.registry else None
        if entry is None:
            return []
        cfg = RETRIEVAL_CONFIG
        out = []
        try:
            from libzim.search import Searcher, Query
            terms = _content_terms(query)
            qterms = set(terms)
            # W2a: several complementary ZIM queries so generic words do not
            # hijack ranking; union the article paths they return.
            zim_queries = []
            if cfg.get("deep_multiquery", True):
                for span in self._entity_spans(query):   # named entities first
                    zim_queries.append(span)
                rare = sorted(terms, key=self._doc_freq)
                if len(rare) >= 2:
                    zim_queries.append(" ".join(rare[:3]))    # discriminating
            if terms:
                zim_queries.append(" ".join(terms[:8]))       # raw content terms
            else:
                zim_queries.append(query)
            seen_q, q_list = set(), []
            for q in zim_queries:
                ql = q.lower().strip()
                if ql and ql not in seen_q:
                    seen_q.add(ql); q_list.append(q)
            searcher = Searcher(entry["archive"])
            per_q = cfg.get("deep_per_query", 3)
            paths, seen_p = [], set()
            for q in q_list:
                try:
                    res = searcher.search(Query().set_query(q))
                    for p in res.getResults(0, per_q):
                        if p not in seen_p:
                            seen_p.add(p); paths.append(p)
                except Exception:
                    continue
            paths = paths[:n_articles]
            for path in paths:
                try:
                    e = entry["archive"].get_entry_by_path(path)
                    raw = bytes(e.get_item().content).decode(
                        'utf-8', errors='ignore')
                    text = self._clean_html(raw)
                    if len(text) < 300:
                        continue
                    words = text.split()
                    if cfg.get("deep_locate", True):
                        pieces = self._article_windows(
                            words, qterms, n_chunks)
                    else:
                        pieces = []
                        for i in range(0, len(words), 120):
                            pc = words[i:i + 160]
                            if len(pc) >= 40:
                                pieces.append(' '.join(pc))
                            if len(pieces) >= n_chunks:
                                break
                    for ct in pieces:
                        out.append({
                            "title":  str(e.title),
                            "text":   ct,
                            "source": entry["label"],
                            "origin": "deep:" + name,
                        })
                except Exception:
                    continue
        except Exception as ex:
            self.last_diag.setdefault("deep_errors", {})[name] = str(ex)
        return out

    def _deep_candidates(self, query, qvec, cfg):
        if self.registry is None:
            return [], []
        out = []
        routed = []
        for name in self.registry.always_archives():
            out += self._search_one_archive(
                name, query, cfg["deep_articles"],
                cfg.get("deep_chunks_wiki", cfg["deep_chunks"]))
        routed_names = set()
        for name, sim in self.registry.route(qvec, k=cfg["route_k"]):
            if sim < cfg["route_floor"]:
                continue
            routed.append((name, round(sim, 2)))
            routed_names.add(name)
            out += self._search_one_archive(
                name, query, cfg["deep_articles_spec"], cfg["deep_chunks"])
        # Force-include need-specific specialists (e.g. skeptics for
        # claimcheck) even if they fell below the routing floor.
        if cfg.get("use_epistemic_fusion"):
            need = _epistemic_need(query)
            for fname in _FORCE_ON_NEED.get(need, []):
                if fname not in routed_names and self.registry.get(fname):
                    routed.append((fname, "forced"))
                    out += self._search_one_archive(
                        fname, query, cfg["deep_articles_spec"],
                        cfg["deep_chunks"])
        return out, routed

    # ── Generator 5: anchors ────────────────────────────────────────

    def _anchor_candidates(self, anchor_titles, n_chunks):
        out = []
        for title in (anchor_titles or [])[:3]:
            try:
                rows = self.db.execute(
                    "SELECT title, text, source FROM passages "
                    "WHERE title = ? LIMIT ?", (title, n_chunks)).fetchall()
                for r in rows:
                    out.append({"title": r["title"], "text": r["text"],
                                "source": r["source"], "origin": "anchor"})
                if not rows and self.registry is not None:
                    for name in self.registry.always_archives():
                        deep = self._search_one_archive(
                            name, title, 1, n_chunks)
                        for c in deep:
                            c["origin"] = "anchor"
                        out.extend(deep)
            except Exception:
                continue
        return out

    # ── Main search ─────────────────────────────────────────────────

    def search(self, query, source_weights=None, expand_fn=None,
               anchor_titles=None):
        if not self.ready:
            return [], 0.0, 0.0
        cfg = RETRIEVAL_CONFIG
        t0 = time.time()
        diag = {"timings": {}}

        qvec = self.embedder.embed_one(query)

        t = time.time()
        title_ids = (self._title_candidates(query, cfg["title_k"])
                     if cfg["use_title"] else [])
        diag["timings"]["title"] = round(time.time() - t, 3)

        t = time.time()
        bm25_ids = (self._bm25_candidates(query, cfg["bm25_k"])
                    if cfg["use_bm25"] else [])
        diag["timings"]["bm25"] = round(time.time() - t, 3)

        t = time.time()
        vec_ids = (self._vector_candidates(qvec, cfg["vector_k"])
                   if cfg["use_vector"] else [])
        diag["timings"]["vector"] = round(time.time() - t, 3)

        origin_of = {}
        for pid in title_ids:
            origin_of.setdefault(pid, "title")
        for pid in bm25_ids:
            origin_of.setdefault(pid, "bm25")
        for pid in vec_ids:
            origin_of.setdefault(pid, "vector")
        pool_ids = list(origin_of.keys())

        candidates = []
        if pool_ids:
            qmarks = ",".join("?" * len(pool_ids))
            rows = self.db.execute(
                f"SELECT id, title, text, source FROM passages "
                f"WHERE id IN ({qmarks})", pool_ids).fetchall()
            meta = {r["id"]: r for r in rows}
            for pid in pool_ids:
                r = meta.get(pid)
                if r is None:
                    continue
                candidates.append({
                    "title": r["title"], "text": r["text"],
                    "source": r["source"], "origin": origin_of[pid],
                })

        if cfg["use_anchors"] and anchor_titles:
            t = time.time()
            candidates += self._anchor_candidates(
                anchor_titles, cfg["anchor_chunks"])
            diag["timings"]["anchor"] = round(time.time() - t, 3)

        if cfg["deep_archive"] == "always":
            t = time.time()
            deep, routed = self._deep_candidates(query, qvec, cfg)
            candidates += deep
            diag["timings"]["deep"] = round(time.time() - t, 3)
            diag["routed"] = routed

        seen, unique = set(), []
        for c in candidates:
            key = (c["title"], c["text"][:80])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        if not unique:
            return [], 0.0, round(time.time() - t0, 3)

        # Passage hygiene: strip nav/boilerplate; drop pure-cruft candidates
        if cfg.get("strip_boilerplate", True):
            _cleaned = []
            for c in unique:
                _ow = len(c["text"].split())
                _st = self._strip_nav(c["text"])
                _sw = len(_st.split())
                if _ow >= 60 and _sw < 25:
                    continue
                if _sw >= 25:
                    c["text"] = _st
                _cleaned.append(c)
            unique = _cleaned
            if not unique:
                return [], 0.0, round(time.time() - t0, 3)

        t = time.time()
        texts = [f"{c['title']} — {c['text']}" for c in unique]
        logits = self.reranker.score(query, texts)
        diag["timings"]["rerank"] = round(time.time() - t, 3)
        for c, lg in zip(unique, logits):
            c["logit"] = lg

        ENCYC = ('wikipedia', 'wikipedia (deep)', 'wikipedia medicine',
                 'wikiversity', 'wikibooks', 'wikinews', 'wikivoyage',
                 'wiktionary', 'wikiquote', 'ifixit')
        ab = cfg['authority_bonus']
        def _rank_key(c):
            bonus = ab if any(c['source'].startswith(p) for p in ENCYC) else 0.0
            return -(c['logit'] + bonus)
        # W4a: medical-source preference — prefer RELEVANT Wikipedia/medical
        # content over folk/forum sources on health queries. Runs regardless of
        # fusion; relevance floor stops noise (irrelevant encyclopedic) rising.
        _need = _epistemic_need(query)
        diag["epistemic_need"] = _need
        if cfg.get("medical_affinity", True) and _need == "medical":
            mb = cfg.get("medical_bonus", 4.0)
            mfloor = cfg.get("medical_floor", 0.0)
            for c in unique:
                is_med = c["source"].startswith("wikipedia")
                c["epi_bonus"] = mb if (is_med and c["logit"] > mfloor) else 0.0
        elif cfg.get("use_epistemic_fusion"):
            want = _NEED_AFFINITY.get(_need, set())
            ab = cfg["epistemic_affinity_bonus"]
            for c in unique:
                et = _label_etype(c["source"], self.registry)
                c["epi_bonus"] = ab if et in want else 0.0
        else:
            for c in unique:
                c["epi_bonus"] = 0.0
        ENCYC2 = ('wikipedia', 'wikipedia (deep)', 'wikipedia medicine',
                  'wikiversity', 'wikibooks', 'wikinews', 'wikivoyage',
                  'wiktionary', 'wikiquote', 'ifixit')
        ab2 = cfg['authority_bonus']
        def _rank_key2(c):
            auth = ab2 if any(c['source'].startswith(p) for p in ENCYC2) else 0.0
            return -(c['logit'] + auth + c.get('epi_bonus', 0.0))
        order = sorted(unique, key=_rank_key2)
        _abl = cfg.get("ablation_sources", "all")
        if _abl == "wiki_pure":
            order = [c for c in order if c["source"].startswith("wikipedia")]
        elif _abl == "wiki_deep":
            order = [c for c in order
                     if c["source"].startswith("wikipedia")
                     or not c["origin"].startswith("deep")]

        final, per_title = [], {}
        for c in order:
            n = per_title.get(c["title"], 0)
            if n >= cfg["max_per_title"]:
                continue
            per_title[c["title"]] = n + 1
            final.append(c)
            if len(final) >= cfg["final_k"]:
                break

        top = final[0]["logit"] if final else -20.0
        all_logits = sorted((c["logit"] for c in order), reverse=True)
        median = (all_logits[len(all_logits) // 2]
                  if all_logits else -20.0)
        confidence = 1.0 / (1.0 + math.exp(-top))

        diag.update({
            "top_logit": round(top, 2),
            "separation": round(top - median, 2),
            "pool_size": len(unique),
            "winners": [(c["title"], c["origin"], round(c["logit"], 2))
                        for c in final],
        })
        # Epistemic-gate signal (consumed by engine only if fusion on)
        epi_need = diag.get("epistemic_need", "factual")
        ENCYC_G = ('wikipedia', 'wikipedia (deep)', 'wikipedia medicine',
                   'wikiversity', 'wikibooks', 'wikinews', 'wikivoyage',
                   'wiktionary', 'wikiquote', 'ifixit')
        # Is a domain-matched specialist among the top-final results,
        # and what is the best logit of any ENCYCLOPEDIC source?
        epi_specialist_in_top = False
        best_encyc_logit = -20.0
        best_specialist_logit = -20.0
        for c in final:
            et = _label_etype(c["source"], self.registry)
            is_encyc = any(c["source"].startswith(p) for p in ENCYC_G)
            if is_encyc:
                best_encyc_logit = max(best_encyc_logit, c["logit"])
            if epi_need == "claimcheck" and et == "debunking":
                epi_specialist_in_top = True
                best_specialist_logit = max(best_specialist_logit, c["logit"])
            elif epi_need == "howto" and et == "instructional" and not is_encyc:
                epi_specialist_in_top = True
                best_specialist_logit = max(best_specialist_logit, c["logit"])
        # keep the old #1-only signal too (back-compat)
        epi_specialist_top = False
        if final:
            top_et = _label_etype(final[0]["source"], self.registry)
            if ((epi_need == "claimcheck" and top_et == "debunking") or
                (epi_need == "howto" and top_et == "instructional")):
                epi_specialist_top = True
        diag["epi_specialist_top"] = epi_specialist_top
        diag["epi_specialist_in_top"] = epi_specialist_in_top
        diag["epi_best_specialist_logit"] = round(best_specialist_logit, 2)
        diag["epi_best_encyc_logit"] = round(best_encyc_logit, 2)
        diag["epi_need"] = epi_need
        self.last_diag = diag

        results = [{
            "title":  c["title"],
            "text":   c["text"],
            "source": c["source"],
            "score":  round(1.0 / (1.0 + math.exp(-c["logit"])), 3),
            "logit":  round(c["logit"], 2),
            "tier":   2 if c["origin"].startswith("deep") else 1,
            "origin": c["origin"],
        } for c in final]

        return results, round(confidence, 3), round(time.time() - t0, 3)
