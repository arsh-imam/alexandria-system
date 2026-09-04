"""
ALEXANDRIA Compute Tier — deterministic dispatch for computable queries.

The model never chooses tools; the SYSTEM detects computable queries
and injects an exact, machine-computed result as ground truth.
High-precision detection only: when unsure, return None and let the
normal pipeline handle it.
"""
import ast
import re
import math
import operator
from datetime import datetime, date

# ── Safe arithmetic (AST whitelist — no eval) ───────────────────────

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.FloorDiv: operator.floordiv}


def _safe_eval(expr):
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("disallowed")
    return walk(ast.parse(expr, mode="eval"))


def _fmt(x):
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            x = round(x)
        else:
            return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:,}"


ARITH_RE = re.compile(
    r"(?:what\s+is|what's|calculate|compute|how\s+much\s+is|evaluate)?\s*"
    r"([\d\s\.\+\-\*/x×÷\(\)\^%]+)\s*[=\?]*\s*$", re.I)


def try_arithmetic(q):
    m = ARITH_RE.match(q.strip())
    if not m:
        return None
    expr = m.group(1).strip()
    if not re.search(r"\d\s*[\+\-\*/x×÷\^%]\s*\d", expr):
        return None          # needs an actual operation
    expr = (expr.replace("x", "*").replace("×", "*")
                .replace("÷", "/").replace("^", "**"))
    expr = re.sub(r"(\d)\s*%", r"(\1/100)", expr)   # trailing percents
    expr = re.sub(r"\s+", "", expr)
    if not re.fullmatch(r"[\d\.\+\-\*/\(\)]+|[\d\.\+\-\*/\(\)]*\*\*[\d\.\+\-\*/\(\)]+", expr):
        if not re.fullmatch(r"[\d\.\+\-\*/\(\)\*]+", expr):
            return None
    try:
        val = _safe_eval(expr)
    except Exception:
        return None
    return f"{m.group(1).strip()} = {_fmt(val)}"


PCT_OF = re.compile(r"(?:what\s+is\s+|what's\s+)?([\d\.]+)\s*(?:%|percent)\s+of\s+([\d,\.]+)", re.I)
X_OF_Y = re.compile(r"what\s+percent(?:age)?\s+(?:is\s+)?([\d,\.]+)\s+of\s+([\d,\.]+)", re.I)


def try_percentage(q):
    m = PCT_OF.search(q)
    if m:
        p = float(m.group(1)); y = float(m.group(2).replace(",", ""))
        return f"{_fmt(p)}% of {_fmt(y)} = {_fmt(p * y / 100)}"
    m = X_OF_Y.search(q)
    if m:
        x = float(m.group(1).replace(",", "")); y = float(m.group(2).replace(",", ""))
        if y == 0:
            return None
        return f"{_fmt(x)} is {_fmt(100 * x / y)}% of {_fmt(y)}"
    return None


UNITS = {
    ("km", "miles"): (0.621371, "miles"), ("miles", "km"): (1.609344, "km"),
    ("kg", "pounds"): (2.204623, "lb"),  ("pounds", "kg"): (0.453592, "kg"),
    ("kg", "lb"): (2.204623, "lb"),      ("lb", "kg"): (0.453592, "kg"),
    ("m", "feet"): (3.28084, "ft"),      ("feet", "m"): (0.3048, "m"),
    ("meters", "feet"): (3.28084, "ft"), ("feet", "meters"): (0.3048, "m"),
    ("cm", "inches"): (0.393701, "in"),  ("inches", "cm"): (2.54, "cm"),
    ("liters", "gallons"): (0.264172, "gal"), ("gallons", "liters"): (3.785412, "l"),
    ("g", "oz"): (0.035274, "oz"),       ("oz", "g"): (28.349523, "g"),
}
ALIASES = {"kilometers": "km", "kilometres": "km", "mile": "miles",
           "kilograms": "kg", "kilogram": "kg", "pound": "pounds", "lbs": "pounds",
           "metres": "m", "meter": "m", "metre": "m", "foot": "feet", "ft": "feet",
           "centimeters": "cm", "centimetres": "cm", "inch": "inches", "in": "inches",
           "litres": "liters", "litre": "liters", "liter": "liters",
           "gallon": "gallons", "grams": "g", "gram": "g", "ounces": "oz", "ounce": "oz"}

CONV_RE = re.compile(r"(?:convert\s+)?([\d\.,]+)\s*([a-z]+)\s+(?:to|in|into)\s+([a-z]+)", re.I)
TEMP_RE = re.compile(r"([\-\d\.]+)\s*(?:degrees?\s*)?(f|fahrenheit|c|celsius)\s+(?:to|in|into)\s+(?:degrees?\s*)?(f|fahrenheit|c|celsius)", re.I)


def try_units(q):
    m = TEMP_RE.search(q)
    if m:
        v = float(m.group(1))
        a = m.group(2)[0].lower(); b = m.group(3)[0].lower()
        if a == b:
            return None
        out = (v - 32) * 5 / 9 if a == "f" else v * 9 / 5 + 32
        return (f"{_fmt(v)}°{a.upper()} = {_fmt(round(out, 2))}°{b.upper()}")
    m = CONV_RE.search(q)
    if m:
        v = float(m.group(1).replace(",", ""))
        a = ALIASES.get(m.group(2).lower(), m.group(2).lower())
        b = ALIASES.get(m.group(3).lower(), m.group(3).lower())
        if (a, b) in UNITS:
            f, label = UNITS[(a, b)]
            return f"{_fmt(v)} {a} = {_fmt(round(v * f, 4))} {label}"
    return None


# ── Date math ───────────────────────────────────────────────────────

DATE_FMTS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%B %d %Y",
             "%d %B %Y", "%b %d, %Y", "%b %d %Y", "%d %b %Y"]


def _parse_date(s):
    s = s.strip().rstrip("?.,")
    for f in DATE_FMTS:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            continue
    return None


BETWEEN_RE = re.compile(r"days?\s+between\s+(.+?)\s+and\s+(.+?)[\?\.]?\s*$", re.I)
WEEKDAY_RE = re.compile(r"(?:what\s+)?day\s+of\s+the\s+week\s+(?:is|was|will\s+be)\s+(.+?)[\?\.]?\s*$", re.I)


def try_dates(q):
    m = BETWEEN_RE.search(q)
    if m:
        d1, d2 = _parse_date(m.group(1)), _parse_date(m.group(2))
        if d1 and d2:
            return f"days between {d1} and {d2} = {abs((d2 - d1).days)} days"
    m = WEEKDAY_RE.search(q)
    if m:
        d = _parse_date(m.group(1))
        if d:
            return f"{d} is a {d.strftime('%A')}"
    return None


# ── Word arithmetic ("3547 plus 9292") ──────────────────────────────

WORD_OP_RE = re.compile(
    r"^(?:what\s+is\s+|what's\s+|whats\s+|calculate\s+|compute\s+|"
    r"how\s+much\s+is\s+)?"
    r"(-?[\d,\.]+)\s+"
    r"(plus|added\s+to|minus|times|multiplied\s+by|divided\s+by)\s+"
    r"(-?[\d,\.]+)\s*[=\?]*\s*$", re.I)
_WORD_OP = {"plus": "+", "added to": "+", "minus": "-",
            "times": "*", "multiplied by": "*", "divided by": "/"}


def try_word_math(q):
    m = WORD_OP_RE.match(q.strip())
    if not m:
        return None
    a = float(m.group(1).replace(",", ""))
    op = re.sub(r"\s+", " ", m.group(2).lower())
    b = float(m.group(3).replace(",", ""))
    sym = _WORD_OP.get(op)
    if not sym:
        return None
    if sym == "/" and b == 0:
        return None
    val = {"+": a + b, "-": a - b, "*": a * b, "/": (a / b if b else 0)}[sym]
    return f"{_fmt(a)} {sym} {_fmt(b)} = {_fmt(val)}"


# ── Time conversions ("how many seconds in 13 minutes") ─────────────

_TIME_S = {"second": 1, "seconds": 1, "sec": 1, "secs": 1,
           "minute": 60, "minutes": 60, "min": 60, "mins": 60,
           "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
           "day": 86400, "days": 86400, "week": 604800, "weeks": 604800,
           "month": 2629800, "months": 2629800,
           "year": 31557600, "years": 31557600}
_TU = (r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|"
       r"months?|years?)")
HOWMANY_TIME_RE = re.compile(
    r"how\s+many\s+" + _TU + r"\s+(?:are\s+)?(?:there\s+)?in\s+"
    r"([\d,\.]+)\s+" + _TU, re.I)
TIMECONV_RE = re.compile(
    r"(?:convert\s+)?([\d,\.]+)\s*" + _TU + r"\s+(?:to|in|into)\s+"
    + _TU, re.I)


def try_time(q):
    m = HOWMANY_TIME_RE.search(q)
    if m:
        to_u = m.group(1).lower(); val = float(m.group(2).replace(",", ""))
        from_u = m.group(3).lower()
        if from_u in _TIME_S and to_u in _TIME_S:
            r = val * _TIME_S[from_u] / _TIME_S[to_u]
            return f"{_fmt(val)} {from_u} = {_fmt(r)} {to_u}"
    m = TIMECONV_RE.search(q)
    if m:
        val = float(m.group(1).replace(",", "")); from_u = m.group(2).lower()
        to_u = m.group(3).lower()
        if from_u in _TIME_S and to_u in _TIME_S and from_u != to_u:
            r = val * _TIME_S[from_u] / _TIME_S[to_u]
            return f"{_fmt(val)} {from_u} = {_fmt(r)} {to_u}"
    return None


# ── Powers & roots ──────────────────────────────────────────────────

SQRT_RE   = re.compile(r"(?:what\s+is\s+|what's\s+)?(?:the\s+)?square\s+root\s+of\s+([\d,\.]+)", re.I)
SQUARED_RE = re.compile(r"(?:what\s+is\s+|what's\s+)?([\d,\.]+)\s+squared\b", re.I)
CUBED_RE   = re.compile(r"(?:what\s+is\s+|what's\s+)?([\d,\.]+)\s+cubed\b", re.I)
POWER_RE   = re.compile(r"(?:what\s+is\s+)?([\d,\.]+)\s+to\s+the\s+power\s+(?:of\s+)?([\d,\.]+)", re.I)


def try_powroot(q):
    m = SQRT_RE.search(q)
    if m:
        v = float(m.group(1).replace(",", ""))
        if v >= 0:
            return f"square root of {_fmt(v)} = {_fmt(round(math.sqrt(v), 6))}"
    m = SQUARED_RE.search(q)
    if m:
        v = float(m.group(1).replace(",", ""))
        return f"{_fmt(v)} squared = {_fmt(v * v)}"
    m = CUBED_RE.search(q)
    if m:
        v = float(m.group(1).replace(",", ""))
        return f"{_fmt(v)} cubed = {_fmt(v * v * v)}"
    m = POWER_RE.search(q)
    if m:
        a = float(m.group(1).replace(",", "")); b = float(m.group(2).replace(",", ""))
        if abs(b) <= 100:
            return f"{_fmt(a)} to the power of {_fmt(b)} = {_fmt(a ** b)}"
    return None


def detect_and_compute(query):
    """Return an exact result string, or None if not computable."""
    q = query.strip()
    if len(q) > 120:
        return None
    for fn in (try_percentage, try_time, try_units, try_dates,
               try_powroot, try_word_math, try_arithmetic):
        try:
            r = fn(q)
        except Exception:
            r = None
        if r:
            return r
    return None
