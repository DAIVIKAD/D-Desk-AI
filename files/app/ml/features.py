"""
Shared ML feature helpers for training and inference.
"""

import re


CATEGORY_KEYWORDS = {
    "Network": [
        "internet", "network", "wifi", "wi-fi", "lan", "vpn", "bandwidth",
        "connection", "router", "switch", "ip", "dns", "ping", "ethernet",
        "firewall", "proxy", "subnet", "gateway", "dhcp", "tcp",
        "latency", "wireless", "access point", "ssid", "port", "cable",
    ],
    "Software": [
        "software", "application", "app", "crash", "install", "outlook",
        "teams", "office", "excel", "word", "error", "update", "license",
        "login", "password", "authentication", "sap", "erp", "browser",
        "chrome", "edge", "windows", "os", "boot", "blue screen", "bsod",
        "freeze", "not responding", "compatibility", "driver", "plugin",
        "antivirus", "malware", "virus", "patch", "configuration",
    ],
    "Hardware": [
        "laptop", "computer", "pc", "monitor", "keyboard", "mouse",
        "memory", "ram", "hard disk", "ssd", "cpu", "fan", "battery",
        "server", "ups", "power supply", "motherboard", "graphics card",
        "usb", "dock", "display", "screen", "charger", "overheating",
    ],
    "Printer": [
        "printer", "print", "scanner", "cartridge", "ink", "toner",
        "paper jam", "offline printer", "hp", "xerox", "printing",
        "spooler", "print queue", "duplex", "colour", "color",
        "laser", "inkjet", "paper tray", "fax", "scan", "photocopy",
    ],
}

URGENCY_WORDS = [
    "urgent", "critical", "immediately", "asap", "emergency",
    "down", "outage", "blocking", "stopped", "crashed",
    "failure", "broken", "dead", "unresponsive", "cannot",
]

NEGATION_PATTERNS = [
    r"\bnot\s+working\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bunable\b",
    r"\bwon't\b",
    r"\bdoesn't\b",
    r"\bdoes\s+not\b",
    r"\bfailed\b",
    r"\bfailing\b",
    r"\bnot\s+responding\b",
    r"\bnot\s+connecting\b",
    r"\bnot\s+loading\b",
    r"\bnot\s+opening\b",
    r"\bno\s+access\b",
    r"\bno\s+connection\b",
    r"\bno\s+internet\b",
]

ENTITY_PATTERNS = [
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    r"\bfloor\s+\d+\b",
    r"\b(?:pc|sw|rt|srv|ws)-[\w-]+\b",
    r"\b(?:port|ext)\s*:?\s*\d+\b",
    r"\b\d+\s*(?:users?|workstations?|systems?)\b",
]

TEMPORAL_PATTERNS = [
    r"\bsince\s+(?:morning|yesterday|last\s+\w+)\b",
    r"\bfor\s+\d+\s*(?:hours?|minutes?|days?)\b",
    r"\bright\s+now\b",
    r"\bpast\s+\d+\b",
    r"\bsince\s+\d+\b",
    r"\ball\s+day\b",
    r"\ball\s+morning\b",
    r"\bwhole\s+(?:day|week|morning)\b",
    r"\bkeeps?\s+(?:happening|recurring|crashing|disconnecting)\b",
]

IMPACT_PATTERNS = [
    r"\bentire\s+(?:floor|department|building|office|team)\b",
    r"\ball\s+(?:users|employees|staff|workstations|systems)\b",
    r"\bwhole\s+(?:floor|department|office|team|network)\b",
    r"\beveryone\b",
    r"\bmultiple\s+(?:users|systems|departments)\b",
    r"\b\d{2,}\s+(?:users?|people|employees?|workstations?)\b",
]

NEGATIVE_WORDS = {
    "frustrated", "angry", "annoyed", "terrible", "awful", "horrible",
    "unacceptable", "ridiculous", "pathetic", "useless", "waste",
    "broken", "ruined", "destroyed", "impossible", "nightmare",
    "painful", "suffering", "struggling", "desperate", "stuck",
    "failing", "failed", "worst", "never", "nothing", "nowhere",
}

EMPHASIS_PATTERNS = [
    r"\b(\w+)\s+\1\b",
    r"\bvery\s+very\b",
    r"\bso\s+so\b",
    r"\bkeeps?\s+\w+ing\b",
    r"\bagain\s+and\s+again\b",
    r"\bover\s+and\s+over\b",
    r"\bstill\s+not\b",
    r"\bstill\s+(?:broken|down|failing)\b",
    r"!{2,}",
]

FACILITY_PATTERNS = [
    r"\ba\/c\b",
    r"\bac not working\b",
    r"\bair\s*condition(?:er|ing)\b",
    r"\bnot cooling\b",
    r"\bcooling issue\b",
    r"\broom is hot\b",
    r"\bcabin is hot\b",
    r"\btemperature issue\b",
]

HIGH_PRIORITY_HINTS = [
    "urgent", "critical", "asap", "emergency", "immediately",
    "down", "outage", "production", "blocking", "cannot work",
    "all users", "entire floor", "everyone", "security breach",
    "compromised", "mission critical", "server down",
]

MEDIUM_PRIORITY_HINTS = [
    "slow", "intermittent", "partial", "degraded", "timeout",
    "disconnect", "flickering", "paper jam", "offline", "draining",
    "freezing", "overheating", "lagging", "error", "issue",
    "problem", "not working", "cannot", "unable", "failed",
]

LOW_PRIORITY_HINTS = [
    "how to", "request", "question", "suggestion", "when possible",
    "no rush", "information", "inquiry", "feedback", "training",
    "guidance", "future reference", "need to install", "need access",
    "replacement needed", "scheduled maintenance", "procurement",
]


def extract_category_features(text: str) -> list:
    text_lower = text.lower()
    words = text_lower.split()
    features = [
        min(len(text) / 500.0, 1.0),
        min(len(words) / 50.0, 1.0),
    ]
    for category in ["Network", "Software", "Hardware", "Printer"]:
        keywords = CATEGORY_KEYWORDS[category]
        features.append(sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1))
    return features


def extract_priority_features(text: str) -> list:
    text_lower = text.lower()
    word_tokens = re.findall(r"\b\w+\b", text_lower)

    urgency_count = sum(1 for word in URGENCY_WORDS if word in text_lower)
    negation_count = sum(len(re.findall(pattern, text_lower)) for pattern in NEGATION_PATTERNS)
    entity_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in ENTITY_PATTERNS)
    temporal_hits = sum(len(re.findall(pattern, text_lower)) for pattern in TEMPORAL_PATTERNS)
    impact_hits = sum(len(re.findall(pattern, text_lower)) for pattern in IMPACT_PATTERNS)
    emphasis_hits = sum(len(re.findall(pattern, text_lower)) for pattern in EMPHASIS_PATTERNS)
    negative_count = sum(1 for word in word_tokens if word in NEGATIVE_WORDS)

    affected_match = re.search(
        r"(\d+)\s*(?:users?|workstations?|systems?|people|employees?)",
        text_lower,
    )
    affected_users = int(affected_match.group(1)) if affected_match else 0

    has_error = 1.0 if re.search(
        r"\b(?:error|err|code|fault)\s*:?\s*(?:0x)?[\da-fA-F]{3,}\b",
        text_lower,
    ) else 0.0

    if word_tokens:
        type_token_ratio = len(set(word_tokens)) / len(word_tokens)
    else:
        type_token_ratio = 0.0

    return [
        min(len(text) / 500.0, 1.0),
        min(urgency_count / 5.0, 1.0),
        min(negation_count / 3.0, 1.0),
        min(entity_count / 5.0, 1.0),
        min(text.count("!") / 3.0, 1.0),
        min(sum(1 for char in text if char.isupper()) / max(len(text), 1), 1.0),
        min(affected_users / 50.0, 1.0),
        min(temporal_hits / 3.0, 1.0),
        min(impact_hits / 2.0, 1.0),
        min(emphasis_hits / 3.0, 1.0),
        min(negative_count / 3.0, 1.0),
        has_error,
        type_token_ratio,
    ]


def infer_priority_label(text: str, fallback: str = "Medium") -> str:
    text_lower = text.lower()

    high_hits = sum(1 for hint in HIGH_PRIORITY_HINTS if hint in text_lower)
    medium_hits = sum(1 for hint in MEDIUM_PRIORITY_HINTS if hint in text_lower)
    low_hits = sum(1 for hint in LOW_PRIORITY_HINTS if hint in text_lower)

    negation_count = sum(len(re.findall(pattern, text_lower)) for pattern in NEGATION_PATTERNS)
    temporal_hits = sum(len(re.findall(pattern, text_lower)) for pattern in TEMPORAL_PATTERNS)
    impact_hits = sum(len(re.findall(pattern, text_lower)) for pattern in IMPACT_PATTERNS)
    emphasis_hits = sum(len(re.findall(pattern, text_lower)) for pattern in EMPHASIS_PATTERNS)

    affected_match = re.search(
        r"(\d+)\s*(?:users?|workstations?|systems?|people|employees?)",
        text_lower,
    )
    affected_users = int(affected_match.group(1)) if affected_match else 0

    if affected_users >= 5:
        medium_hits += 1
    if affected_users >= 15:
        high_hits += 1

    if negation_count >= 1:
        medium_hits += 1
    if negation_count >= 2:
        high_hits += 1

    if temporal_hits >= 1:
        medium_hits += 1
    if temporal_hits >= 2:
        high_hits += 1

    if impact_hits >= 1:
        high_hits += 1
    if impact_hits >= 2:
        high_hits += 1

    if emphasis_hits >= 1 or text.count("!") >= 2:
        high_hits += 1

    if "how to" in text_lower or "just a question" in text_lower:
        low_hits += 2

    if high_hits >= 2:
        return "High"
    if low_hits >= 2 and high_hits == 0 and negation_count == 0:
        return "Low"
    if medium_hits >= 1:
        return "Medium"
    if fallback in {"High", "Low"}:
        return fallback
    return "Medium"


def is_facilities_issue(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in FACILITY_PATTERNS)
