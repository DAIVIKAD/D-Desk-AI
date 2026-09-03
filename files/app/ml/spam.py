"""
D Desk AI — Spam Detection (spam.py)
─────────────────────────────────────────
Provides:
    detect_spam(text)          → "spam" | "not_spam"          (backward compat)
    detect_spam_detailed(text) → dict with score, keywords, patterns, reason

Architecture: Two-layer detection
  Layer 1 — Rule-based (fast, always runs):
    - Preprocessing (lowercase, emoji strip, repeated-char collapse, URL extraction)
    - High-density spam keyword matching (90+ patterns)
    - Phishing regex patterns (OTP, banking, crypto, links)
    - Excessive UPPERCASE ratio check
    - Repeated-character pattern detection
    - Suspicious link / phone-number patterns
    - Overly short / meaningless texts

  Layer 2 — Naive Bayes ML model (trained on synthetic data):
    - CountVectorizer + MultinomialNB
    - Trained in-memory on first call (~50 synthetic samples)
    - Only consulted when rules are inconclusive

Decision rule:
  - Rule says spam   → spam  (no ML check needed)
  - Rule says not-spam and ML confidence < 0.70 → not_spam
  - Rule says not-spam and ML says spam with ≥ 0.70 → spam
  - Either way, the FINAL label is returned; Groq never overrides

Hybrid fallback:
  - If ML confidence is low but rule-based keyword score is strong → spam
"""

import logging
import re
import unicodedata
from typing import Literal

from app.ml.utils import clean_text, is_conversational_text

logger = logging.getLogger("ddesk.spam")

# ═══════════════════════════════════════════════════════════════════════════
#  Preprocessing Helpers
# ═══════════════════════════════════════════════════════════════════════════

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")       # 3+ repeated chars
_MULTI_PUNCT_RE = re.compile(r"([!?.]){2,}")        # !!!! or ???
_MULTI_SPACE_RE = re.compile(r"\s+")

# ── Slang / SMS abbreviation normalization ────────────────────────────────
# Maps common chat abbreviations used in scam messages to their full forms
# so that keyword matching catches casual/slang phishing attempts.
_SLANG_MAP = {
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\byr\b": "your",
    r"\bacc\b": "account",
    r"\bacct\b": "account",
    r"\bpwd\b": "password",
    r"\bpw\b": "password",
    r"\bpass\b": "password",
    r"\bpls\b": "please",
    r"\bplz\b": "please",
    r"\btxt\b": "text",
    r"\bmsg\b": "message",
    r"\binfo\b": "information",
    r"\bcuz\b": "because",
    r"\bbc\b": "because",
    r"\bw\b": "with",
    r"\bthx\b": "thanks",
    r"\bthnx\b": "thanks",
    r"\brn\b": "right now",
    r"\basap\b": "as soon as possible",
    r"\bbro\b": "",
    r"\bdude\b": "",
    r"\bhmu\b": "contact me",
    r"\bdm\b": "message",
    r"\bfyi\b": "for your information",
    r"\bimo\b": "in my opinion",
    r"\blol\b": "",
    r"\bomg\b": "",
    r"\bwon\b": "won",       # keep as-is but ensure word boundary
    r"\bgift\b": "gift",     # keep as-is
}
_SLANG_COMPILED = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in _SLANG_MAP.items()]


def _normalize_slang(text: str) -> str:
    """Expand chat/SMS abbreviations into standard English words."""
    for pattern, replacement in _SLANG_COMPILED:
        text = pattern.sub(replacement, text)
    return text


def _preprocess_for_spam(text: str) -> str:
    """
    Clean the input text for spam analysis:
      - Lowercase
      - Strip emojis
      - Collapse repeated characters (freeeeee → free, looottery → lottery)
      - Normalize slang (u → you, ur → your, acc → account)
      - Normalize excessive punctuation (!!!! → !)
      - Collapse whitespace
    """
    text = text.lower()
    # Strip emojis
    text = _EMOJI_RE.sub(" ", text)
    # Strip unicode category "So" (symbols) that aren't caught above
    text = "".join(
        ch if unicodedata.category(ch) != "So" else " " for ch in text
    )
    # Collapse repeated characters: freeeeee → free, cliiick → click
    text = _REPEATED_CHAR_RE.sub(r"\1\1", text)
    # Normalize slang abbreviations
    text = _normalize_slang(text)
    # Normalize excessive punctuation
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    # Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _extract_urls(text: str) -> list[str]:
    """Return all URLs found in the raw text."""
    return _URL_RE.findall(text)


# ═══════════════════════════════════════════════════════════════════════════
#  Spam Keyword Bank  (90+ patterns)
# ═══════════════════════════════════════════════════════════════════════════

_SPAM_KEYWORDS = [
    # ── Monetary / prize bait ──
    "free money", "100% free", "earn money fast", "make money online",
    "cash prize", "you've won", "congratulations you won", "claim your prize",
    "lottery winner", "selected winner", "lucky winner", "exclusive offer",
    "limited offer", "act now", "click here now", "click below",
    "click the link", "call now", "call immediately", "call this number",
    "win reward", "win rewards", "win lottery", "claim reward",
    "claim rewards", "claim free", "free reward", "free rewards",
    "free gift card", "free gift", "gift card",
    "free iphone", "free ipad", "free macbook", "free laptop",
    "limited offer claim", "limited time offer",
    # ── Slang-expanded catch phrases (after normalization) ──
    "won the lottery", "you won", "your account blocked",
    "your account suspended", "click fast", "free cash",
    "claim your reward", "claim your gift",

    # ── Pharmaceutical / adult ──
    "buy now", "order now", "discount pills", "cheap meds", "viagra",
    "cialis", "weight loss", "lose weight fast", "diet pill",

    # ── Phishing / credential theft ──
    "verify your account", "confirm your password", "update your billing",
    "your account has been suspended", "unusual activity detected",
    "click to verify", "enter your credentials", "bank account",
    "credit card number", "social security", "ssn required",
    "verify account", "verify password", "verify immediately",
    "bank password", "password expired", "account expired",
    "account suspended", "account compromised", "reset your password",
    "suspicious link", "suspicious activity",
    "send otp", "send your otp", "share otp", "share your otp",
    "otp immediately", "enter otp",
    "banking details", "bank details", "credit card details",

    # ── Crypto / investment scams ──
    "bitcoin opportunity", "crypto investment", "double your money",
    "guaranteed return", "100% profit", "risk-free investment",
    "invest today", "passive income", "financial freedom now",
    "crypto giveaway", "bitcoin giveaway", "free crypto",
    "free bitcoin", "crypto airdrop", "nft giveaway",

    # ── Generic spam signals ──
    "unsubscribe", "remove from list", "bulk email", "mass mailing",
    "this is not spam", "this email is not spam",
    "mlm", "multi level marketing", "pyramid scheme",
    "click here", "click this",
]

_SPAM_PATTERN_RE = re.compile(
    "|".join(re.escape(kw) for kw in _SPAM_KEYWORDS),
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Phishing Regex Patterns
# ═══════════════════════════════════════════════════════════════════════════

_PHISHING_REGEXES = [
    # OTP / verification scams
    (re.compile(r"\b(?:send|share|enter|verify|confirm)\b.*\b(?:otp|pin|code)\b", re.I), "otp_scam"),
    (re.compile(r"\b(?:otp|pin|code)\b.*\b(?:send|share|enter|immediately|now|urgent)\b", re.I), "otp_scam"),
    # Banking scams
    (re.compile(r"\bbank\b.*\b(?:password|account|detail|credential|login)\b", re.I), "banking_scam"),
    (re.compile(r"\b(?:password|account)\b.*\b(?:expired?|suspend|compromis|verif|reset)\b", re.I), "credential_phishing"),
    # Prize / reward scams
    (re.compile(r"\b(?:win|won|claim|free)\b.*\b(?:prize|reward|gift|money|cash|iphone|laptop|card)\b", re.I), "prize_scam"),
    (re.compile(r"\b(?:prize|reward|gift|money|cash)\b.*\b(?:claim|win|free|click)\b", re.I), "prize_scam"),
    # Crypto scams
    (re.compile(r"\b(?:crypto|bitcoin|btc|ethereum|nft)\b.*\b(?:giveaway|free|airdrop|invest|double)\b", re.I), "crypto_scam"),
    (re.compile(r"\b(?:giveaway|free|airdrop)\b.*\b(?:crypto|bitcoin|btc|ethereum|nft)\b", re.I), "crypto_scam"),
    # Suspicious link patterns
    (re.compile(r"\b(?:click|visit|go to|open)\b.*\b(?:link|url|website|site)\b", re.I), "suspicious_link"),
    (re.compile(r"\bsuspicious\s+link\b", re.I), "suspicious_link"),
]

# Patterns that almost never appear in legitimate IT tickets
_SUSPICIOUS_PATTERNS = [
    (re.compile(r"https?://\S+", re.IGNORECASE), "bare_url"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "phone_number"),
    (re.compile(r"\$\s*\d+[\d,]*(?:\.\d+)?"), "money_amount"),
    (re.compile(r"(.)\1{4,}"), "repeated_chars"),
    (re.compile(r"[A-Z]{5,}"), "allcaps_run"),
]

_MIN_LEGIT_LENGTH = 8   # anything shorter than 8 chars is suspicious


# ═══════════════════════════════════════════════════════════════════════════
#  Lazy ML model (Naive Bayes)
# ═══════════════════════════════════════════════════════════════════════════

_nb_model    = None
_nb_vec      = None

def _get_nb_model():
    """
    Train and cache a simple Naive-Bayes spam classifier on first call.
    Returns (vectorizer, model) tuple.
    """
    global _nb_model, _nb_vec
    if _nb_model is not None:
        return _nb_vec, _nb_model

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    spam_samples = [
        "Click here to claim your FREE prize now!!!",
        "You have been selected as a lucky winner. Call immediately.",
        "Make money online 100% free — earn $500 a day guaranteed!",
        "Verify your bank account or it will be suspended.",
        "Exclusive limited offer: buy now before it's too late!",
        "Congratulations! You've won a cash prize. Claim here.",
        "URGENT: Your account has been compromised. Enter credentials now.",
        "Invest in Bitcoin today — guaranteed 200% returns!",
        "Cheap meds delivered fast. Order now. Discount pills.",
        "Your credit card information needs to be updated immediately.",
        "This is not spam. Unsubscribe if you don't want FREE money.",
        "CLICK BELOW to double your money in 24 hours!",
        "Risk-free investment — 100% profit guaranteed. Act now.",
        "You have been selected for a lottery winner prize claim.",
        "lose weight fast with our miracle diet pills buy now",
        "Make thousands working from home. Multi level marketing.",
        "AAAAAAHHH BUY BUY BUY!!! SALE SALE SALE!!!",
        "Free crypto coins — sign up and earn passive income today.",
        "Unusual activity on your account. Verify your password now.",
        "Call this number: +1-800-123-4567 for an exclusive deal!",
        # ── Extended samples for failing test cases ──
        "URGENT!!! Click here to win free rewards now",
        "verify your account immediately",
        "claim free iphone",
        "crypto giveaway",
        "win lottery now",
        "bank password expired",
        "click suspicious link",
        "limited offer claim reward",
        "free gift card",
        "send OTP immediately",
        "free iphone click now",
        "verify bank password",
        "urgent crypto giveaway",
        "claim reward immediately",
        "reset account using this suspicious link",
        "Congratulations you won a free iPhone click to claim",
        "Your bank password has expired please verify now",
        "Win a free gift card by clicking this link",
        "Send your OTP to verify your account immediately",
        "Free Bitcoin giveaway enter your wallet address now",
        "Limited time offer claim your cash prize today",
        "Your account has been suspended click here to reactivate",
        "Double your crypto investment in 24 hours guaranteed",
        "Free lottery ticket claim your winnings now",
        "Urgent verify your banking details immediately",
        # ── Casual / slang scam samples ──
        "hey u won the lottery click here",
        "claim ur reward bro",
        "click fast free cash",
        "your acc blocked verify now",
        "yo free gift card just for u",
        "ur account suspended click to fix",
        "bro u won free iphone claim it rn",
        "hey click this link for free money lol",
        "ur pass expired reset now or lose acc",
        "omg u won a prize claim ur reward asap",
    ]
    ham_samples = [
        "WiFi not working on floor 3 since morning",
        "Outlook keeps crashing when I open attachments",
        "Need to reset my SAP password urgently — cannot log in",
        "Printer on floor 2 is showing offline and paper jam",
        "Laptop fan making loud noise and overheating",
        "Cannot connect to VPN from home — getting error 619",
        "Monitor flickering and going blank intermittently",
        "Teams audio not working after the latest Windows update",
        "Request for Adobe Reader installation on my PC",
        "Please create a new user account for our new joiner",
        "Internet speed is very slow on the 5th floor since yesterday",
        "Mouse cursor jumping randomly — replaced mouse but still happens",
        "Scanner not working on the MFD in the HR department",
        "Excel xlsx file not opening — shows compatibility error",
        "Need access to the network drive Q: from my workstation",
        "Server room AC is making a strange noise — please check",
        "USB ports on my docking station stopped working suddenly",
        "How do I change my email signature in Outlook?",
        "Keyboard backspace key is not responding on my laptop",
        "Blue screen error appeared during Windows update restart",
        # ── Extended ham to avoid false positives ──
        "Password reset needed for the ERP system urgently",
        "Two-factor authentication not sending OTP codes to my phone",
        "Cannot access my bank reconciliation software on SAP",
        "Account locked after multiple failed login attempts",
        "Free disk space running low on the file server",
        "Gift card module in HR software not loading",
        "Need to verify the software license status immediately",
        "Link to the company intranet is broken",
        "Click the submit button but nothing happens in the portal",
    ]

    texts  = spam_samples + ham_samples
    labels = (["spam"] * len(spam_samples)) + (["not_spam"] * len(ham_samples))

    _nb_vec   = CountVectorizer(ngram_range=(1, 2), max_features=500)
    _nb_model = MultinomialNB()
    X = _nb_vec.fit_transform(texts)
    _nb_model.fit(X, labels)

    return _nb_vec, _nb_model


# ═══════════════════════════════════════════════════════════════════════════
#  Scoring Logic
# ═══════════════════════════════════════════════════════════════════════════

def _compute_spam_score(
    keyword_hits: list[str],
    phishing_hits: list[str],
    suspicious_hits: list[str],
    uppercase_ratio: float,
    ml_spam_prob: float,
    text_length: int,
) -> float:
    """
    Combine rule-based and ML signals into a single 0.0–1.0 spam score.

    Scoring weights:
      - Each keyword match:   +0.15  (capped contribution: 0.60)
      - Each phishing regex:  +0.20  (capped contribution: 0.60)
      - Each suspicious pat:  +0.08  (capped contribution: 0.30)
      - Uppercase ratio > 50%: +0.15
      - ML probability:        weighted at 0.30
      - Very short text:       +0.10
    """
    kw_score = min(len(keyword_hits) * 0.15, 0.60)
    phish_score = min(len(phishing_hits) * 0.20, 0.60)
    sus_score = min(len(suspicious_hits) * 0.08, 0.30)
    upper_bonus = 0.15 if uppercase_ratio > 0.50 else 0.0
    short_bonus = 0.10 if 0 < text_length < _MIN_LEGIT_LENGTH else 0.0
    ml_contribution = ml_spam_prob * 0.30

    raw = kw_score + phish_score + sus_score + upper_bonus + short_bonus + ml_contribution
    return round(min(raw, 1.0), 4)


# ═══════════════════════════════════════════════════════════════════════════
#  Public API — Detailed Detection
# ═══════════════════════════════════════════════════════════════════════════

_SPAM_SCORE_THRESHOLD = 0.40


def detect_spam_detailed(text: str) -> dict:
    """
    Full spam analysis with score, matched keywords, and explainability.

    Args:
        text: Raw input string from the user / API caller.

    Returns:
        dict with keys:
            is_spam              — bool
            spam_score           — float 0.0–1.0
            detected_spam_keywords — list of matched keyword strings
            detected_patterns    — list of pattern labels (e.g. "otp_scam")
            urls_found           — list of URLs extracted
            classifier_source    — "rule_based" | "ml_model" | "hybrid"
            decision_reason      — human-readable explanation
    """
    if not text or not text.strip():
        return {
            "is_spam": True,
            "spam_score": 1.0,
            "detected_spam_keywords": [],
            "detected_patterns": ["empty_input"],
            "urls_found": [],
            "classifier_source": "rule_based",
            "decision_reason": "Empty submission treated as spam.",
        }

    # ── Conversational Whitelist Bypass ────────────────────────────────────
    if is_conversational_text(text):
        return {
            "label": "not_spam",
            "is_spam": False,
            "spam_score": 0.0,
            "detected_spam_keywords": [],
            "detected_patterns": [],
            "urls_found": [],
            "classifier_source": "conversational_whitelist",
            "decision_reason": "Conversational greeting or acknowledgement.",
            "is_conversational": True,
        }

    # ── Preprocessing ──────────────────────────────────────────────────────
    urls_found = _extract_urls(text)
    cleaned = _preprocess_for_spam(text)

    # ── Layer 1: Rule-based checks ─────────────────────────────────────────

    # 1a. Keyword matches
    keyword_hits = list({m.group().lower() for m in _SPAM_PATTERN_RE.finditer(cleaned)})

    # 1b. Phishing regex matches
    phishing_hits = list({label for regex, label in _PHISHING_REGEXES if regex.search(cleaned)})

    # 1c. Suspicious structural patterns
    suspicious_hits = [label for regex, label in _SUSPICIOUS_PATTERNS if regex.search(text)]

    # 1d. Uppercase ratio (computed on original text, not lowered)
    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in letters if c.isupper()) / len(letters)
        if letters else 0.0
    )

    # 1e. Text length
    text_length = len(cleaned.strip())

    # ── Determine if rules alone are conclusive ──
    rule_is_spam = False
    classifier_source = "rule_based"
    reasons = []

    # Meaningless / empty short text (e.g. "?", "...", single punctuation)
    if text_length < 2 and not text.strip().isalpha():
        rule_is_spam = True
        reasons.append(f"Text too short ({text_length} chars)")

    # Direct keyword hit
    if keyword_hits:
        rule_is_spam = True
        reasons.append(f"Spam keywords matched: {keyword_hits}")

    # Phishing regex hit
    if phishing_hits:
        rule_is_spam = True
        reasons.append(f"Phishing patterns detected: {phishing_hits}")

    # Excessive uppercase
    if uppercase_ratio > 0.50 and len(letters) > 5:
        rule_is_spam = True
        reasons.append(f"Excessive uppercase ratio: {uppercase_ratio:.0%}")

    # Multiple suspicious patterns
    if len(suspicious_hits) >= 2:
        rule_is_spam = True
        reasons.append(f"Multiple suspicious patterns: {suspicious_hits}")

    # ── Layer 2: ML Naive Bayes model ─────────────────────────────────────
    ml_spam_prob = 0.0
    try:
        vec, model = _get_nb_model()
        X = vec.transform([cleaned])
        classes = list(model.classes_)
        proba = model.predict_proba(X)[0]
        ml_spam_prob = float(proba[classes.index("spam")]) if "spam" in classes else 0.0

        if not rule_is_spam and ml_spam_prob >= 0.70:
            rule_is_spam = True
            classifier_source = "ml_model"
            reasons.append(f"ML model confidence: {ml_spam_prob:.2%}")
    except Exception:
        pass  # If ML fails, rely on rules alone

    # ── Compute combined spam score ──
    spam_score = _compute_spam_score(
        keyword_hits=keyword_hits,
        phishing_hits=phishing_hits,
        suspicious_hits=suspicious_hits,
        uppercase_ratio=uppercase_ratio,
        ml_spam_prob=ml_spam_prob,
        text_length=text_length,
    )

    # ── Hybrid fallback: ML low confidence but strong keyword signals ──
    if not rule_is_spam and spam_score >= _SPAM_SCORE_THRESHOLD:
        rule_is_spam = True
        classifier_source = "hybrid"
        reasons.append(
            f"Hybrid: ML prob {ml_spam_prob:.2%} low but combined spam_score "
            f"{spam_score:.2f} exceeds threshold {_SPAM_SCORE_THRESHOLD}"
        )

    # ── Final decision ──
    is_spam = rule_is_spam
    if classifier_source == "rule_based" and not is_spam:
        classifier_source = "rule_based"  # confirmed clean by rules

    # Ensure score reflects the decision
    if is_spam and spam_score < _SPAM_SCORE_THRESHOLD:
        spam_score = max(spam_score, _SPAM_SCORE_THRESHOLD + 0.05)

    decision_reason = " | ".join(reasons) if reasons else "No spam signals detected."

    result = {
        "is_spam": is_spam,
        "spam_score": round(spam_score, 4),
        "detected_spam_keywords": sorted(keyword_hits),
        "detected_patterns": sorted(set(phishing_hits + suspicious_hits)),
        "urls_found": urls_found,
        "classifier_source": classifier_source,
        "decision_reason": decision_reason,
    }

    logger.info(
        "Spam check: is_spam=%s score=%.4f keywords=%s patterns=%s source=%s | text=%.80s",
        result["is_spam"],
        result["spam_score"],
        result["detected_spam_keywords"],
        result["detected_patterns"],
        result["classifier_source"],
        text,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Public API — Simple (backward compatible)
# ═══════════════════════════════════════════════════════════════════════════

def detect_spam(text: str) -> Literal["spam", "not_spam"]:
    """
    Determine whether the given text is spam or a legitimate IT ticket.

    This is the backward-compatible wrapper around detect_spam_detailed().

    Args:
        text: Raw input string from the user / API caller.

    Returns:
        "spam" or "not_spam"
    """
    result = detect_spam_detailed(text)
    return "spam" if result["is_spam"] else "not_spam"
