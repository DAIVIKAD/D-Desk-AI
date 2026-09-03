"""
D Desk AI — ML Training Script
────────────────────────────────────
Standalone CLI script for training the ticket classifier.

Train the model:
    python app/ml/train.py

Test a prediction against the saved model:
    python app/ml/train.py --test "wifi not working"

Trains:
    - TF-IDF vectorizer  (unigram + bigram + trigram, 8 000 features)
    - Best category model chosen from Logistic Regression / SVM / Random Forest
    - Logistic Regression for priority  (High / Medium / Low)

Saves to app/ml/:
    - model.pkl       (dict with both classifiers + encoders)
    - vectorizer.pkl  (dict with both vectorizers)

The saved models are in standard scikit-learn .pkl format and are
portable across machines (GPU training server → local Mac M1).
"""

import os
import sys
import pickle
import argparse
import random
import warnings

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

def load_real_dataset():
    """
    Load real dataset from app/ml/data.csv when available.
    Expected helpdesk columns:
        subject, body, type, priority

    Returns:
        [(text, label, priority), ...]
    """
    import pandas as pd

    data_path = os.path.join(_HERE, "data.csv")

    if not os.path.exists(data_path):
        print("        No real dataset found at app/ml/data.csv; using synthetic data only.")
        return []

    try:
        df = pd.read_csv(data_path)
        print(f"        Read {len(df)} rows from data.csv")

        columns = {str(column).strip().lower(): column for column in df.columns}
        text_columns = [name for name in ("subject", "body") if name in columns]
        missing_text_columns = [name for name in ("subject", "body") if name not in columns]
        type_column = columns.get("type")
        priority_column = columns.get("priority")

        if not text_columns:
            print("        Warning: data.csv is missing both 'subject' and 'body' columns.")
            return []
        if missing_text_columns:
            print(
                "        Warning: missing text columns in data.csv: "
                + ", ".join(missing_text_columns)
                + ". Using available fields only."
            )
        if type_column is None:
            print("        Warning: data.csv is missing required column: type")
            return []
        if priority_column is None:
            print("        Warning: missing column 'priority' in data.csv; defaulting to Medium.")

        def clean_cell(value):
            if pd.isna(value):
                return ""
            return str(value).strip()

        def normalize_text(*parts):
            combined = " ".join(part for part in parts if part)
            return " ".join(combined.lower().split())

        def normalize_category(value):
            normalized = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
            if not normalized:
                return ""
            if any(token in normalized for token in ("printer", "print", "scan", "copier", "toner")):
                return "Printer"
            if any(token in normalized for token in ("network", "internet", "wifi", "wi fi", "vpn", "lan", "dns", "router", "switch")):
                return "Network"
            if any(token in normalized for token in ("hardware", "laptop", "desktop", "keyboard", "mouse", "screen", "monitor", "battery", "charger", "device")):
                return "Hardware"
            if any(token in normalized for token in ("software", "application", "app", "browser", "outlook", "windows", "email", "login")):
                return "Software"
            if normalized == "other":
                return "Other"
            return "Other"

        def normalize_priority(value):
            normalized = " ".join(value.lower().split())
            if normalized == "high":
                return "High"
            if normalized == "low":
                return "Low"
            if normalized == "medium":
                return "Medium"
            return "Medium"

        corpus = []
        skipped_rows = 0
        for _, row in df.iterrows():
            subject = clean_cell(row[columns["subject"]]) if "subject" in columns else ""
            body = clean_cell(row[columns["body"]]) if "body" in columns else ""
            label = normalize_category(clean_cell(row[type_column]))
            priority = normalize_priority(clean_cell(row[priority_column])) if priority_column else "Medium"
            text = normalize_text(subject, body)

            if not text or not label:
                skipped_rows += 1
                continue
            corpus.append((text, label, priority))

        print(f"        Loaded {len(corpus)} usable rows from data.csv")
        if skipped_rows:
            print(f"        Skipped {skipped_rows} empty or invalid rows from data.csv")
        return corpus

    except Exception as exc:
        print(f"        Failed to load app/ml/data.csv: {exc}")
        return []
# ── Add project root to sys.path so this script runs from any CWD ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.ml.features import (
    extract_category_features,
    extract_priority_features,
    infer_priority_label,
)
from app.ml.utils import (
    get_model_path,
    predict_label_and_confidence,
    preprocess_prediction_text,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic Training Dataset
# ═══════════════════════════════════════════════════════════════════════════

# NEW CODE START
def _build_multilingual_samples():
    """
    Add balanced multilingual support examples without removing or replacing
    the existing synthetic corpus.
    """
    return [
        ("wifi kaam nahi kar raha", "Network", "High"),
        ("internet kaam nahi kar raha", "Network", "High"),
        ("vpn kaam nahi kar raha", "Network", "High"),
        ("network kaam nahi kar raha", "Network", "High"),
        ("wifi kelasa madthilla", "Network", "High"),
        ("internet kelasa madthilla", "Network", "High"),
        ("vpn connect agalla", "Network", "High"),
        ("lan kelasa madthilla", "Network", "High"),
        ("wifi velai seyyala", "Network", "High"),
        ("internet velai seyyala", "Network", "High"),
        ("vpn velai seyyala", "Network", "High"),
        ("network velai seyyala", "Network", "High"),
        ("outlook kaam nahi kar raha", "Software", "Medium"),
        ("excel kaam nahi kar raha", "Software", "Medium"),
        ("software kaam nahi kar raha", "Software", "Medium"),
        ("login kaam nahi kar raha", "Software", "High"),
        ("outlook kelasa madthilla", "Software", "Medium"),
        ("software open agalla", "Software", "Medium"),
        ("excel kelasa madthilla", "Software", "Medium"),
        ("login agalla", "Software", "High"),
        ("outlook velai seyyala", "Software", "Medium"),
        ("software velai seyyala", "Software", "Medium"),
        ("excel velai seyyala", "Software", "Medium"),
        ("login velai seyyala", "Software", "High"),
        ("printer kaam nahi kar raha", "Printer", "Medium"),
        ("scanner kaam nahi kar raha", "Printer", "Medium"),
        ("network printer kaam nahi kar raha", "Printer", "High"),
        ("office printer kaam nahi kar raha", "Printer", "Medium"),
        ("printer kelasa madthilla", "Printer", "Medium"),
        ("scanner kelasa madthilla", "Printer", "Medium"),
        ("print queue agalla", "Printer", "Medium"),
        ("office printer kelasa madthilla", "Printer", "Medium"),
        ("printer velai seyyala", "Printer", "Medium"),
        ("scanner velai seyyala", "Printer", "Medium"),
        ("print velai seyyala", "Printer", "Medium"),
        ("office printer velai seyyala", "Printer", "Medium"),
    ]
# NEW CODE END


def generate_training_corpus():
    """
    Generate a rich synthetic IT support corpus (~1 500 labelled samples).
    Identical domain coverage to the in-memory classifier but decoupled so
    the saved model can be retrained independently.
    """
    random.seed(42)
    corpus = []

    network_templates = [
        "Internet is very slow on floor {floor}",
        "No internet connection on my workstation",
        "WiFi keeps disconnecting in the conference room",
        "VPN connection drops every {n} minutes",
        "Cannot access the network drive",
        "LAN cable not working at desk {desk}",
        "Bandwidth is extremely low today",
        "DNS resolution failing for internal sites",
        "Network switch on floor {floor} seems down",
        "Ping to server shows high latency and packet loss",
        "Internet not working for {n} users on floor {floor}",
        "Router restart needed on floor {floor}",
        "Proxy server blocking legitimate websites",
        "VPN authentication keeps failing",
        "Network connectivity issues affecting entire department",
        "Wireless access point not broadcasting SSID",
        "IP address conflict on subnet 192.168.{floor}.x",
        "Firewall blocking application traffic",
        "DHCP not assigning IP addresses",
        "Ethernet port on wall plate not active",
        "Slow internet speed affecting all workstations",
        "Network cable damaged near server room",
        "Cannot connect to corporate WiFi network",
        "Gateway unreachable from floor {floor}",
        "TCP connection timeouts to application server",
        "Network outage affecting building B since morning",
        "VLAN configuration issue causing connectivity drops",
        "NAT table overflow on edge router",
        "Traceroute shows high hops to internal server",
        "DNS lookup returning wrong IP for intranet portal",
        "WiFi signal very weak in basement floor {floor}",
        "Network port on switch SW-{floor}A is flapping",
        "IPsec tunnel to remote site keeps dropping",
        "NTP sync failing across all network devices",
    ]

    software_templates = [
        "MS Teams not launching after latest update",
        "Outlook crashes when opening attachments",
        "Excel cannot open xlsx files shows compatibility error",
        "Windows blue screen error on restart",
        "Software license expired for AutoCAD",
        "Chrome browser not loading any pages",
        "Cannot login to SAP application",
        "Application shows error code {code} on startup",
        "Word document formatting is corrupted",
        "Need to install Adobe Reader on my system",
        "Operating system update failed midway",
        "Antivirus scan detected malware threats",
        "Browser keeps redirecting to wrong pages",
        "Software installation stuck at {n} percent",
        "Teams meeting audio not working after update",
        "Password reset needed for the ERP system",
        "Application freezes when processing large files",
        "Edge browser not compatible with internal portal",
        "Office 365 activation error on new laptop",
        "Boot failure after Windows patch update",
        "Login authentication failing for all applications",
        "Driver update caused display issues",
        "Plugin not loading in browser",
        "Configuration error in email client settings",
        "Software patch causing system instability",
        "DLL missing error when launching accounting software",
        "Runtime error in SAP transaction {code}",
        "Registry corruption after failed uninstall",
        "Service pack installation keeps rolling back",
        "Java runtime not found by internal web application",
        "Two-factor authentication not sending OTP codes",
        "PowerShell execution policy blocking automation script",
        "Windows Defender quarantined a critical DLL file",
        "Application shows access denied error code {code}",
    ]

    hardware_templates = [
        "Laptop fan making loud grinding noise",
        "Monitor screen flickering and going black",
        "Keyboard keys not responding properly",
        "Mouse cursor jumping around the screen",
        "Computer running extremely slow with high CPU usage",
        "Laptop battery draining in {n} minutes",
        "USB ports not detecting any devices",
        "Docking station not connecting to external monitor",
        "Hard disk making clicking sounds",
        "RAM insufficient for running required applications",
        "Power supply unit making buzzing noise",
        "Laptop overheating and shutting down automatically",
        "Desktop not powering on at all",
        "Screen display has dead pixels",
        "Laptop charger not charging the battery",
        "Workstation freezing under heavy load",
        "Server UPS battery replacement needed",
        "Processor usage at 100 percent constantly",
        "SSD showing health warnings",
        "External display not detected by laptop",
        "Graphics card causing display artifacts",
        "Motherboard diagnostic LED showing error",
        "Computer hangs during startup",
        "Slow performance due to low memory",
        "Noisy fan on desktop computer",
        "Laptop hinge is broken and screen is loose",
        "Webcam not detected after BIOS update",
        "Touchpad gestures not working on new firmware",
        "Headset audio crackling through USB connection",
        "Server blade showing amber fault LED",
        "RAID array degraded on storage server",
        "CPU thermal throttling under normal workload",
        "NVMe drive not visible in BIOS after firmware update",
        "Peripheral devices disconnecting randomly from USB hub",
    ]

    printer_templates = [
        "HP LaserJet showing offline on floor {floor}",
        "Printer paper jam in tray 2",
        "Toner cartridge needs replacement",
        "Print queue stuck with {n} pending jobs",
        "Scanner not working on the multifunction device",
        "Printer printing blank pages",
        "Duplex printing not working properly",
        "Color printer only printing in black and white",
        "Print spooler service keeps stopping",
        "Cannot add network printer from my workstation",
        "Xerox machine showing error code {code}",
        "Printer ink levels critically low",
        "Fax machine not sending documents",
        "Photocopier glass needs cleaning",
        "Paper tray sensor malfunction on printer",
        "Laser printer producing faded output",
        "Inkjet printer head clogged",
        "Printer driver not compatible with OS",
        "Scan to email feature not working",
        "Copier paper feed roller needs replacement",
        "Label printer not calibrating correctly",
        "Plotter producing distorted large-format prints",
        "Secure print jobs not releasing at the device",
        "Printer showing supply level error but toner is new",
    ]

    other_templates = [
        "How to set up email on mobile phone",
        "Request for new employee account creation",
        "Need access card for server room",
        "Suggestion for improving IT portal interface",
        "Question about data backup procedures",
        "Information needed about software procurement",
        "Request for ergonomic assessment of desk setup",
        "How to join the video conference bridge",
        "Need training on new HR management system",
        "General inquiry about IT policies",
        "How do I change my email signature",
        "Request to add shared mailbox access",
        "Need guidance on file sharing best practices",
        "When is the next scheduled maintenance window",
        "Feedback on the new ticketing system interface",
    ]

    # ── Base corpus ──
    for tmpl in network_templates:
        for _ in range(5):
            text = tmpl.format(floor=random.randint(1, 10), desk=random.randint(100, 900), n=random.randint(3, 30))
            corpus.append((text, "Network", random.choice(["High", "Medium", "Medium", "Medium"])))

    for tmpl in software_templates:
        for _ in range(5):
            text = tmpl.format(code=random.randint(1000, 9999), n=random.randint(10, 95))
            corpus.append((text, "Software", random.choice(["Medium", "Medium", "Low", "Medium"])))

    for tmpl in hardware_templates:
        for _ in range(5):
            text = tmpl.format(n=random.randint(10, 120))
            corpus.append((text, "Hardware", random.choice(["Medium", "Low", "Medium", "Medium"])))

    for tmpl in printer_templates:
        for _ in range(5):
            text = tmpl.format(floor=random.randint(1, 8), n=random.randint(5, 60), code=random.randint(100, 999))
            corpus.append((text, "Printer", random.choice(["Low", "Medium", "Low", "Low"])))

    for tmpl in other_templates:
        for _ in range(5):
            corpus.append((tmpl, "Other", "Low"))

    # ── High-priority prefixed variants ──
    high_prefixes = [
        "URGENT: ", "CRITICAL: ", "All users affected — ",
        "Production system down — ", "Emergency: ", "Cannot work — ",
        "Entire floor affected — ", "ASAP needed — ",
    ]
    for prefix in high_prefixes:
        for tmpl in network_templates[:8]:
            corpus.append((prefix + tmpl.format(floor=random.randint(1, 10), desk=100, n=5), "Network", "High"))
        for tmpl in software_templates[:6]:
            corpus.append((prefix + tmpl.format(code=random.randint(1000, 9999), n=50), "Software", "High"))
        for tmpl in hardware_templates[:5]:
            corpus.append((prefix + tmpl.format(n=random.randint(5, 30)), "Hardware", "High"))
        for tmpl in printer_templates[:3]:
            corpus.append((prefix + tmpl.format(floor=random.randint(1, 6), n=20, code=500), "Printer", "High"))

    # ── Medium prefixes ──
    medium_prefixes = ["Intermittent issue: ", "Sometimes: ", "Partial outage: ", "Affecting a few users: "]
    for prefix in medium_prefixes:
        for tmpl in network_templates[:5]:
            corpus.append((prefix + tmpl.format(floor=random.randint(1, 8), desk=200, n=10), "Network", "Medium"))
        for tmpl in software_templates[:5]:
            corpus.append((prefix + tmpl.format(code=random.randint(1000, 9999), n=40), "Software", "Medium"))

    # ── Low prefixes ──
    low_prefixes = ["When possible: ", "Minor issue: ", "Just a question: ", "No rush but: "]
    for prefix in low_prefixes:
        for tmpl in other_templates[:5]:
            corpus.append((prefix + tmpl, "Other", "Low"))
        for tmpl in software_templates[:3]:
            corpus.append((prefix + tmpl.format(code=1234, n=50), "Software", "Low"))

    # NEW CODE START
    corpus.extend(_build_multilingual_samples())
    # NEW CODE END

    return corpus


def _stabilize_priority_labels(corpus):
    """
    Replace noisy or placeholder priorities with deterministic labels inferred
    from the ticket text. This keeps the synthetic generator unchanged while
    producing more learnable targets for the priority model.
    """
    stabilized = []
    relabeled = 0

    for text, category, priority in corpus:
        inferred_priority = infer_priority_label(text, fallback=priority)
        if inferred_priority != priority:
            relabeled += 1
        stabilized.append((text, category, inferred_priority))

    return stabilized, relabeled


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def train_and_save():
    """Train the classifier and save model + vectorizer pkl files."""
    print("\n" + "═" * 56)
    print("  D Desk AI — ML Training Script")
    print("─" * 56)

    # 1. Generate corpus
    print("  [1/5] Generating training corpus …")
    synthetic_corpus = generate_training_corpus()
    real_data = load_real_dataset()
    corpus = list(synthetic_corpus)

    print(f"        {len(synthetic_corpus)} synthetic samples generated")
    print(f"        {len(real_data)} real samples loaded")

    if real_data:
        corpus.extend(real_data)

    corpus, relabeled_count = _stabilize_priority_labels(corpus)
    random.Random(42).shuffle(corpus)

    print(f"        Total dataset size: {len(corpus)} samples")
    if relabeled_count:
        print(f"        Priority labels normalized for {relabeled_count} samples")

    texts = [c[0] for c in corpus]
    categories = [c[1] for c in corpus]
    priorities = [c[2] for c in corpus]

    # 2. Encoders
    print("  [2/5] Fitting label encoders …")
    cat_enc = LabelEncoder()
    pri_enc = LabelEncoder()
    cat_labels = cat_enc.fit_transform(categories)
    pri_labels = pri_enc.fit_transform(priorities)

    # 3. Vectorizers
    print("  [3/5] Fitting TF-IDF vectorizers …")
    cat_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=8000,
                              stop_words="english", sublinear_tf=True)
    pri_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=5000,
                              stop_words="english", sublinear_tf=True)
    X_cat_tfidf = cat_vec.fit_transform(texts)
    X_pri_tfidf = pri_vec.fit_transform(texts)

    # 4. Engineered features
    X_cat_extra = csr_matrix(np.array([extract_category_features(t) for t in texts]))
    X_pri_extra = csr_matrix(np.array([extract_priority_features(t) for t in texts]))

    X_cat_full = hstack([X_cat_tfidf, X_cat_extra], format="csr")
    X_pri_full = hstack([X_pri_tfidf, X_pri_extra], format="csr")

    # 5. Train
    print("  [4/5] Training category model candidates …")
    # NEW CODE START
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    category_model_factories = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=2000,
            C=1.5,
            solver="lbfgs",
            class_weight="balanced",
        ),
        "SVM": lambda: LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = {}
    for model_name, factory in category_model_factories.items():
        accuracy = cross_val_score(factory(), X_cat_full, cat_labels, cv=cv, scoring="accuracy").mean()
        trained_model = factory()
        trained_model.fit(X_cat_full, cat_labels)
        results[model_name] = (trained_model, accuracy)

    print("        📊 Model Comparison:")
    for model_name, (_, accuracy) in results.items():
        print(f"        {model_name}: {accuracy:.2%}")

    best_model_name = max(results, key=lambda x: results[x][1])
    best_model, best_acc = results[best_model_name]
    print(f"        🏆 Best Model: {best_model_name} ({best_acc:.2%})")

    print("        Training priority model …")
    # NEW CODE END
    pri_model = LogisticRegression(max_iter=2000, C=1.5, solver="lbfgs", class_weight="balanced")

    pri_model.fit(X_pri_full, pri_labels)

    # Cross-validation accuracy
    pri_cv = cross_val_score(pri_model, X_pri_full, pri_labels, cv=cv, scoring="accuracy").mean()

    # 6. Save
    print("  [5/5] Saving model artifacts …")
    model_path = get_model_path("model.pkl")
    vec_path   = get_model_path("vectorizer.pkl")

    model_bundle = {
        "category_model": best_model,
        # NEW CODE START
        "model_name": best_model_name,
        # NEW CODE END
        "priority_model":  pri_model,
        "category_encoder": cat_enc,
        "priority_encoder": pri_enc,
        # NEW CODE START
        "all_results": {model_name: accuracy for model_name, (_, accuracy) in results.items()},
        # NEW CODE END
    }
    vec_bundle = {
        "category_vectorizer": cat_vec,
        "priority_vectorizer": pri_vec,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vec_bundle, f)

    print("═" * 56)
    print(f"  ✅ Category accuracy (CV-3): {best_acc:.2%}")
    print(f"  ✅ Priority accuracy  (CV-3): {pri_cv:.2%}")
    print(f"  📦 Saved → {model_path}")
    print(f"  📦 Saved → {vec_path}")
    print("═" * 56 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Test Mode
# ═══════════════════════════════════════════════════════════════════════════

def test_prediction(text: str):
    """Load saved models and predict on the given text."""
    model_path = get_model_path("model.pkl")
    vec_path   = get_model_path("vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("\n  ⚠️  No saved model found. Run without --test flag first to train.\n")
        return

    with open(model_path, "rb") as f:
        models = pickle.load(f)
    with open(vec_path, "rb") as f:
        vecs = pickle.load(f)

    cat_vec   = vecs["category_vectorizer"]
    pri_vec   = vecs["priority_vectorizer"]
    cat_model = models["category_model"]
    pri_model = models["priority_model"]
    cat_enc   = models["category_encoder"]
    pri_enc   = models["priority_encoder"]
    # NEW CODE START
    model_name = models.get("model_name", "Logistic Regression")
    processed_text = preprocess_prediction_text(text)
    # NEW CODE END

    X_cat = hstack([
        # NEW CODE START
        cat_vec.transform([processed_text]),
        csr_matrix(np.array([extract_category_features(processed_text)])),
        # NEW CODE END
    ], format="csr")
    X_pri = hstack([
        # NEW CODE START
        pri_vec.transform([processed_text]),
        csr_matrix(np.array([extract_priority_features(processed_text)])),
        # NEW CODE END
    ], format="csr")

    # NEW CODE START
    category, cat_conf = predict_label_and_confidence(cat_model, X_cat, cat_enc)
    priority, pri_conf = predict_label_and_confidence(pri_model, X_pri, pri_enc)
    # NEW CODE END

    print("\n" + "═" * 50)
    print("  D Desk AI — Prediction Test")
    print("─" * 50)
    print(f"  Input     : {text}")
    # NEW CODE START
    if processed_text != text.lower().strip():
        print(f"  Processed : {processed_text}")
    print(f"  Model     : {model_name}")
    # NEW CODE END
    print(f"  Category  : {category}  ({cat_conf:.0%} confidence)")
    print(f"  Priority  : {priority}  ({pri_conf:.0%} confidence)")
    print("═" * 50 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="D Desk AI — ML Training & Testing CLI"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        metavar="TEXT",
        help="Test a prediction using the saved model (e.g. --test 'wifi not working')",
    )
    args = parser.parse_args()

    if args.test:
        test_prediction(args.test)
    else:
        train_and_save()
