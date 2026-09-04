"""Markdown -> Qt rich-text HTML (dependency-free) for PyQt6 QLabel display.
Live-streaming friendly. No LaTeX (Qt has no math engine) — simple unicode
fallbacks only. Display-only; does not affect the engine or retrieval."""
import re
import html

_SUP = {"0":"\u2070","1":"\u00b9","2":"\u00b2","3":"\u00b3","4":"\u2074",
        "5":"\u2075","6":"\u2076","7":"\u2077","8":"\u2078","9":"\u2079",
        "n":"\u207f","i":"\u2071","+":"\u207a","-":"\u207b"}

def _unicode_math(s):
    s = re.sub(r"\\times\b", "\u00d7", s)
    s = re.sub(r"\\div\b", "\u00f7", s)
    s = re.sub(r"\\sqrt\b", "\u221a", s)
    s = re.sub(r"\\pi\b", "\u03c0", s)
    s = re.sub(r"\\alpha\b", "\u03b1", s)
    s = re.sub(r"\\beta\b", "\u03b2", s)
    s = re.sub(r"\\mu\b", "\u03bc", s)
    s = re.sub(r"\\Delta\b", "\u0394", s)
    s = re.sub(r"\^([0-9n i\+\-])", lambda m: _SUP.get(m.group(1), "^" + m.group(1)), s)
    return s.replace("$", "")

def _inline(s):
    s = html.escape(s)
    s = _unicode_math(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"__(.+?)__", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+?)`",
               r'<code style="background:#22222a;padding:1px 4px;border-radius:4px;">\1</code>', s)
    return s

def md_to_qthtml(text):
    if not text:
        return ""
    out = []
    in_ul = in_ol = False
    for raw in text.split("\n"):
        st = raw.strip()
        if not st:
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            out.append("<br>")
            continue
        if (st.startswith("Sources:") or st.startswith("Sources (")
                or st.startswith("Note:") or "answered from model knowledge" in st):
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            out.append('<div style="color:#8a8a94;font-size:11px;margin-top:8px;">'
                       + _inline(st) + "</div>")
            continue
        m = re.match(r"(#{1,4})\s+(.*)", st)
        if m:
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            out.append("<b>" + _inline(m.group(2)) + "</b><br>")
            continue
        m = re.match(r"\d+[\.\)]\s+(.*)", st)
        if m:
            if in_ul: out.append("</ul>"); in_ul = False
            if not in_ol: out.append("<ol>"); in_ol = True
            out.append("<li>" + _inline(m.group(1)) + "</li>")
            continue
        m = re.match(r"[-*\u2022]\s+(.*)", st)
        if m:
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append("<li>" + _inline(m.group(1)) + "</li>")
            continue
        if in_ul: out.append("</ul>"); in_ul = False
        if in_ol: out.append("</ol>"); in_ol = False
        out.append(_inline(st) + "<br>")
    if in_ul: out.append("</ul>")
    if in_ol: out.append("</ol>")
    return "".join(out)
