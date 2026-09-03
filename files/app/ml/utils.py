"""
D Desk AI — ML Utilities
─────────────────────────────
Shared helpers used across the ml/ sub-package.

- get_ml_dir()          → absolute path to app/ml/
- get_model_path(name)  → full path to a model artifact inside app/ml/
- clean_text(text)      → lowercase + strip excess whitespace
- normalize_text(text)  → map common multilingual issue phrases
- translate_to_english(text) → best-effort translation for inference
- preprocess_prediction_text(text) → normalize + translate for prediction
"""

import os
import re

import numpy as np

# NEW CODE START
try:
    from googletrans import Translator
except Exception:  # pragma: no cover - graceful fallback when dependency is absent
    Translator = None


_TRANSLATOR = None

_NORMALIZATION_MAP = {
    "kaam nahi kar raha": "not working",
    "kelasa madthilla": "not working",
    "velai seyyala": "not working",
    "kaam nahi": "not working",
    "madthilla": "not working",
    "agalla": "not working",
    "seyyala": "not working",
}
# NEW CODE END


def get_ml_dir() -> str:
    """Return the absolute path to the app/ml/ directory (this file's directory)."""
    return os.path.dirname(os.path.abspath(__file__))


def get_model_path(filename: str) -> str:
    """
    Return the absolute path to a model artifact stored inside app/ml/.

    Args:
        filename: e.g. 'model.pkl', 'vectorizer.pkl', 'image_model.h5'

    Returns:
        Absolute file path string.
    """
    return os.path.join(get_ml_dir(), filename)


def clean_text(text: str) -> str:
    """
    Normalise raw text for ML inference:
      - Lowercase
      - Collapse multiple whitespace runs into single space
      - Strip leading/trailing whitespace

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# NEW CODE START
def normalize_text(text: str) -> str:
    """
    Replace common multilingual support phrases with stable English hints
    before feature extraction or translation.
    """
    normalized = clean_text(text)
    for source_phrase in sorted(_NORMALIZATION_MAP, key=len, reverse=True):
        normalized = normalized.replace(source_phrase, _NORMALIZATION_MAP[source_phrase])
    return clean_text(normalized)


def _get_translator():
    global _TRANSLATOR

    if Translator is None:
        return None
    if _TRANSLATOR is None:
        _TRANSLATOR = Translator()
    return _TRANSLATOR


def translate_to_english(text: str) -> str:
    """
    Translate input text to English during inference.
    Falls back to the original text if translation is unavailable or fails.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return cleaned

    translator = _get_translator()
    if translator is None:
        return cleaned

    try:
        translated = translator.translate(cleaned, dest="en")
        translated_text = getattr(translated, "text", "") or cleaned
        return clean_text(translated_text)
    except Exception:
        return cleaned


def preprocess_prediction_text(text: str) -> str:
    """
    Prediction-time text pipeline:
        raw text -> normalize -> translate -> clean
    """
    normalized = normalize_text(text)
    translated = translate_to_english(normalized)
    return clean_text(translated)


def predict_label_and_confidence(model, X, label_encoder):
    """
    Predict a label and a confidence-like score for models with or without
    predict_proba support.
    """
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        winning_index = int(np.argmax(probabilities[0]))
        encoded_label = int(model.classes_[winning_index])
        confidence = float(probabilities[0][winning_index])
    elif hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(X), dtype=float)
        if decision_scores.ndim == 0:
            decision_scores = np.array([[-decision_scores, decision_scores]], dtype=float)
        elif decision_scores.ndim == 1:
            if decision_scores.size == 1:
                score = float(decision_scores[0])
                decision_scores = np.array([[-score, score]], dtype=float)
            else:
                decision_scores = decision_scores.reshape(1, -1)

        stabilized_scores = decision_scores - decision_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(stabilized_scores)
        probabilities = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)
        winning_index = int(np.argmax(probabilities[0]))
        encoded_label = int(model.classes_[winning_index])
        confidence = float(probabilities[0][winning_index])
    else:
        encoded_label = int(model.predict(X)[0])
        confidence = 1.0

    label = str(label_encoder.inverse_transform([encoded_label])[0])
    return label, confidence


CONVERSATIONAL_PHRASES = {
    "hi", "hello", "hey", "hey there", "hi there", "hello there", "hello team",
    "good morning", "good afternoon", "good evening", "morning", "evening",
    "ok", "okay", "sure", "cool", "fine", "alright", "all right", "k", "kk",
    "yes", "yep", "yeah", "no", "nope", "not yet",
    "thanks", "thank you", "thx", "thank u", "many thanks", "appreciated",
    "bye", "goodbye", "see you", "cya", "take care",
    "how are you", "how are u", "what can you do", "who are you",
    "help", "help me", "support", "test", "testing",
}


ISSUE_INDICATORS = {
    "error", "crash", "broke", "broken", "fail", "failed", "freeze", "frozen",
    "jam", "jammed", "stuck", "offline", "slow", "down", "bug", "issue", "problem",
    "disconnect", "not working", "cannot", "cant", "wont", "doesnt work", "restart",
    "damage", "smoke", "flicker", "burn", "leak", "crack", "spill", "unreachable",
    "spooler", "adapter", "screen", "display", "cable", "wifi", "ethernet", "router",
    "printer", "battery", "laptop", "monitor", "mouse", "keyboard", "vpn", "outlook",
    "password", "login", "locked", "portal", "teams", "excel", "software", "hardware"
}


def is_conversational_text(text: str) -> bool:
    """Check if the text is a conversational greeting, acknowledgement, or small talk."""
    if not text or not text.strip():
        return True
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return True

    # Direct match
    if cleaned in CONVERSATIONAL_PHRASES:
        return True

    has_issue = any(ind in cleaned for ind in ISSUE_INDICATORS)

    # Courtesy / Thanks phrases without IT issues
    if any(p in cleaned for p in ["thank", "thanks", "thx", "appreciate", "great", "awesome", "perfect", "good day", "nice day", "take care"]):
        if not has_issue:
            return True

    words = cleaned.split()
    first_word = words[0] if words else ""

    # Common greetings without technical complaints
    if first_word in {"hi", "hello", "hey", "hola", "greetings"}:
        if not has_issue:
            return True

    # Acknowledgements without technical complaints
    if first_word in {"ok", "okay", "sure", "cool", "fine", "alright", "got it", "understood", "done", "yes", "no", "nope", "yep", "yeah"}:
        if not has_issue:
            return True

    # Time-based greetings
    if cleaned.startswith("good ") and any(t in cleaned for t in ["morning", "afternoon", "evening", "night"]):
        if not has_issue:
            return True

    # General queries about assistant
    if any(q in cleaned for q in ["what can you do", "who are you", "how are you", "what are you", "are you an ai", "how does this work", "can you help me"]):
        if not has_issue:
            return True

    return False
# NEW CODE END
