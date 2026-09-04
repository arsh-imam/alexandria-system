"""
LCATRS Router v2.5 — compositional trigger, calibrated.

v2.4 over-narrowed the exemption (killed real multi-hop). v2.5:
  - SINGLE_ENTITY exempts only true bios/definitions ("who was X",
    "what is X", optionally "and what is X known/famous for")
  - COMPOSITIONAL recognises action-on-entity clauses without
    needing a specific finale-noun list
"""
import re

WRITING_PATTERNS = [
    r'\bwrite (me )?(a|an|the|this)\b', r'\bdraft\b', r'\bcompose\b',
    r'\b(email|cover letter|poem|story|essay|caption|slogan|speech|bio|tweet|haiku)\b.*\b(write|draft|make|create|give|help)\b',
    r'\b(write|draft|make|create|give|help).*\b(email|cover letter|poem|story|essay|caption|slogan|speech|bio|tweet|haiku)\b',
    r'\brewrite\b', r'\brephrase\b', r'\bparaphrase\b', r'\breword\b',
    r'\bsummari[sz]e\b', r'\bshorten\b', r'\bcondense\b', r'\bexpand this\b',
    r'\bmake (this|it|me)\b', r'\bturn (this|these|it)\b',
    r'\bfix (the )?(grammar|spelling|punctuation)\b', r'\bproofread\b',
    r'\btranslate\b', r'\bsound more\b', r'\bmore (formal|casual|polite|concise|confident|professional)\b',
    r'\bbrainstorm\b', r'\b(give me|suggest|generate|come up with) .*(ideas|names|options|titles|slogans|lines)\b',
    r'\bideas? for\b', r'\bnames? for\b', r'\bsubject lines?\b',
    r'\bicebreakers?\b',
    r'\bplan (a|an|my|the)\b', r'\bmake me a\b', r'\bcreate a (schedule|plan|routine|itinerary|list)\b',
    r'\boutline\b', r'\bhelp me (plan|organi[sz]e|structure)\b',
    r'\b(study|meal|workout|morning|weekly) (schedule|plan|routine)\b',
]

CHITCHAT_PATTERNS = [
    r'^(hi|hello|hey|yo|sup|good (morning|afternoon|evening))\b',
    r'^(thanks|thank you|thx|ok|okay|cool|nice|great)\b.{0,20}$',
    r'\bhow are you\b', r'\bwho are you\b', r'\bwhat can you do\b',
]

DETAIL_PATTERNS = [
    r'\bin detail\b', r'\bdetailed\b', r'\bexplain (fully|thoroughly)\b',
    r'\bstep[- ]by[- ]step\b', r'\bcomprehensive\b', r'\beverything about\b',
]

BRIEF_PATTERNS = [
    r'\bbriefly\b', r'\bin short\b', r'\bone (line|sentence)\b',
    r'\bquick(ly)?\b', r'\btl;?dr\b',
]

INFO_GUARDS = [
    r'\bwhat should i (see|do|visit)\b', r'\bbest (time|place|way) to\b',
    r'\bthings to do\b', r'\bis it (safe|worth|better)\b',
    r'\bwhat (is|are|was|were)\b', r'\bhow (does|do|did|is|are)\b',
    r'\bwhy (is|are|do|does|did)\b', r'\bwho (is|was|are|were)\b',
    r'\bfamous\b', r'\bexamples? of\b',
]


# True bio/definition: "who/what is/was X", optionally followed by
# a clarifying clause about that SAME X ("and what is he known for").
SINGLE_ENTITY = re.compile(
    r'^(who|what)\s+(was|is|are|were)\s+\S+(\s+\S+){0,4}'
    r'(\s+and\s+(what|how)\s+(is|are|was|were)\s+'
    r'(he|she|it|they|his|her|its|their|the\s+\S+)\s+\w+'
    r'(\s+\w+){0,3})?\s*\??\s*$',
    re.I)

# Compositional: bridge entities, comparisons, quantitative bridges,
# OR two distinct WH-clauses joined by "and".
COMPOSITIONAL_PATTERNS = [
    # bridge: "the X that/which/who VERB"
    r'\bthe\s+\w+\s+(that|which|who|whose|where)\s+\w+',
    r'\bcompare\b.+\b(and|with|versus|vs)\b',
    r'\b(difference|similarit\w+)\s+between\s+.+\s+and\s+',
    r'\bwho\s+(was|is)\s+(older|younger|taller|richer|first|bigger)\b.*\bor\b',
    r'\bwhich\s+(came|was|is|did)\s+\w*\s*(first|earlier|older|bigger|taller)\b.*\bor\b',
    r'\bhow\s+many\s+(years|days|months)\s+(passed\s+|)between\b',
    r'\btaller\s+than\s+.+\s+(stacked|combined)\b',
    # two distinct WH-clauses
    r'\b(which|what|who|why|how)\s+\w+.+\band\s+(what|which|who|why|how)\s+\w+',
]


def classify_query(query):
    q = query.strip().lower()

    skip = False
    qtype = "general"
    info_locked = any(re.search(p, q) for p in INFO_GUARDS)

    if any(re.search(p, q) for p in CHITCHAT_PATTERNS):
        qtype, skip = "chitchat", True
    elif not info_locked and any(re.search(p, q) for p in WRITING_PATTERNS):
        qtype, skip = "writing", True

    depth = "simple"
    if any(re.search(p, q) for p in DETAIL_PATTERNS):
        depth = "detailed"
    elif any(re.search(p, q) for p in BRIEF_PATTERNS):
        depth = "brief"

    compositional = False
    if not skip and not SINGLE_ENTITY.match(q):
        compositional = any(re.search(p, q) for p in COMPOSITIONAL_PATTERNS)

    return {
        "query_type": qtype,
        "skip_retrieval": skip,
        "depth": depth,
        "compositional": compositional,
        "source_weights": {},
    }
