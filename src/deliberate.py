"""
LCATRS Deliberate Mode v6 — rescore + hardened decision parsing.

v6 rejects malformed SEARCH queries (prompt echoes like "missing
fact", queries with no concrete noun, or near-duplicates) so a bad
decision STOPS the loop cleanly instead of searching garbage. The
final rescore against the original question is retained.
"""
import re

_BAD_SQ = re.compile(
    r"missing fact|most important|the single|<|>|specific thing|"
    r"required fact|unverified", re.I)


def _ev_text(evidence, limit=6, words=60):
    lines = []
    for r in evidence[:limit]:
        lines.append(f"[{r['title']}] "
                     + ' '.join(r['text'].split()[:words]))
    return "\n".join(lines)


def _bad_query(sq, used):
    if not sq or len(sq.split()) < 2 or len(sq.split()) > 14:
        return True
    if _BAD_SQ.search(sq):
        return True
    if sq.lower() in (u.lower() for u in used):
        return True
    if not re.search(r"[a-z]{3,}", sq, re.I):
        return True
    return False


VERIFY_PROMPT = (
    "Question: {q}\n\nEvidence collected so far:\n{ev}\n\n"
    "Think step by step:\n"
    "- What specific facts does the question require? (Watch for "
    "FIRST, oldest, tallest, exact dates, and names that must be "
    "found via another name.)\n"
    "- Is each required fact ACTUALLY stated in the evidence? A "
    "different year, edition, or near-match does NOT count.\n\n"
    "After reasoning, end with ONE final line in EXACTLY this form:\n"
    "DECISION: ENOUGH\n"
    "or\n"
    "DECISION: SEARCH <a short query naming the actual missing "
    "person, place, work, date, or thing>{nothink}")


def run_deliberate(llm, retriever, query, is_qwen3,
                   status_cb=None, max_cycles=3):
    evidence, seen, queries_used, trace = [], set(), [query], []

    def add(results):
        for r in results:
            key = (r["title"], r["text"][:80])
            if key not in seen:
                seen.add(key)
                evidence.append(r)

    res, _, _ = retriever.search(query)
    add(res)
    trace.append({"c": 1, "q": query, "top": [r["title"] for r in res[:3]]})

    for cycle in range(2, max_cycles + 1):
        prompt = VERIFY_PROMPT.format(
            q=query, ev=_ev_text(evidence),
            nothink=(" /no_think" if is_qwen3 else ""))
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=160, temperature=0.0, stream=False)
            text = re.sub(r"<think>.*?</think>", "",
                          out["choices"][0]["message"]["content"],
                          flags=re.S).strip()
        except Exception:
            break
        m = re.search(r"DECISION:\s*(ENOUGH|SEARCH\s+(.+))", text, re.I)
        if not m or m.group(1).upper().startswith("ENOUGH"):
            trace.append({"c": cycle,
                          "decision": (m.group(0)[:80] if m else "none")})
            break
        sq = m.group(2).strip().strip('"').strip(".")[:120]
        if _bad_query(sq, queries_used):
            trace.append({"c": cycle, "decision": "rejected: " + sq[:50]})
            break
        queries_used.append(sq)
        if status_cb:
            status_cb("Looking up: " + sq[:42] + "...", "#d4900a")
        res, _, _ = retriever.search(sq)
        add(res)
        trace.append({"c": cycle, "q": sq,
                      "top": [r["title"] for r in res[:3]]})

    if evidence:
        texts = [f"{r['title']} — {r['text']}" for r in evidence]
        logits = retriever.reranker.score(query, texts)
        for r, lg in zip(evidence, logits):
            r["logit"] = round(lg, 2)
            r["score"] = round(1.0 / (1.0 + 2.718281828 ** (-lg)), 3)
        evidence.sort(key=lambda r: -r["logit"])
    return evidence[:5], queries_used, trace
