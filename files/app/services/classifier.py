"""
D Desk AI — Advanced Multi-Feature Hybrid ML Classifier v4.0
────────────────────────────────────────────────────────────────
Production-grade IT ticket classifier combining TF-IDF vectorisation,
Logistic Regression, and a rich set of hand-crafted features with
domain-specific keyword boosting.

Architecture:
  1. TF-IDF Vectorization (unigram + bigram + trigram) via scikit-learn
  2. Logistic Regression for category & priority prediction
  3. Extended engineered features (v4 — 13 new features total):
     ── Category Features ──
     - Text length (normalised)
     - Word count (normalised)
     - Keyword match density per category
     - IT entity density
     - Sentence count
     - Average word length
     - Special character density
     - Question mark presence
     - Category-specific bigram hit count
     - Technical jargon density
     - Verb-like / Action word ratio

     ── Priority Features ──
     - Text length (normalised)
     - Urgency word count
     - Negation pattern count
     - IT entity density
     - Exclamation count
     - Uppercase ratio
     - Affected user count
     - Temporal urgency signals
     - Impact scope detector
     - Repeated emphasis / frustration signals
     - Negative sentiment density
     - Error code presence
     - Word diversity (Type-Token Ratio)

  4. Keyword confidence boosting for domain-specific accuracy
  5. Real confidence scores via predict_proba + calibration
  6. Explainability — top contributing features per prediction

The model is bootstrapped with a synthetic IT support training corpus
(~1500 samples) generated from keyword templates at initialisation.
"""

import re
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


class EnhancedTicketClassifier:
    """
    Advanced IT ticket classifier v4.0 — combines ML predictions
    with domain-specific rule boosting and full explainability.
    """

    # ═══════════════════════════════════════════════════════════════════
    #  Domain Keyword Dictionaries
    # ═══════════════════════════════════════════════════════════════════

    CATEGORY_KEYWORDS = {
        "Network": [
            "internet", "network", "wifi", "wi-fi", "lan", "vpn", "bandwidth",
            "connection", "router", "switch", "ip", "dns", "ping", "slow internet",
            "ethernet", "firewall", "proxy", "subnet", "gateway", "dhcp",
            "tcp", "packet loss", "latency", "disconnects", "no connectivity",
            "wireless", "access point", "ssid", "port", "cable", "wan",
            "routing", "nat", "vlan", "ipsec", "traceroute", "nslookup",
        ],
        "Software": [
            "software", "application", "app", "crash", "install", "outlook",
            "teams", "office", "excel", "word", "error", "update", "license",
            "login", "password", "authentication", "sap", "erp", "browser",
            "chrome", "edge", "windows", "os", "boot", "blue screen", "bsod",
            "freeze", "not responding", "compatibility", "driver", "plugin",
            "antivirus", "malware", "virus", "patch", "configuration",
            "registry", "dll", "runtime", "service pack", "activation",
        ],
        "Hardware": [
            "laptop", "computer", "pc", "monitor", "keyboard", "mouse",
            "memory", "ram", "hard disk", "ssd", "cpu", "fan", "battery",
            "freeze", "hang", "slow laptop", "server", "ups", "power supply",
            "motherboard", "graphics card", "usb", "dock", "docking station",
            "display", "screen", "charger", "overheating", "noise",
            "workstation", "desktop", "processor", "bios", "firmware",
            "peripheral", "headset", "webcam", "touchpad", "hinge",
        ],
        "Printer": [
            "printer", "print", "scanner", "cartridge", "ink", "toner",
            "paper jam", "offline printer", "hp", "xerox", "printing",
            "spooler", "print queue", "duplex", "colour", "color",
            "laser", "laserjet", "inkjet", "paper tray", "fax",
            "scan", "photocopy", "copier", "plotter", "label printer",
        ],
        "Other": [],
    }

    PRIORITY_KEYWORDS = {
        "High": [
            "urgent", "critical", "down", "not working", "stopped",
            "immediately", "all users", "production", "cannot work",
            "emergency", "outage", "server down", "complete failure",
            "total", "entire", "everyone", "asap", "blocking",
            "data loss", "security breach", "compromised", "deadline",
            "escalate", "showstopper", "mission critical",
        ],
        "Medium": [
            "slow", "intermittent", "sometimes", "issue", "problem",
            "degraded", "partial", "occasional", "affecting", "multiple",
            "few users", "workaround", "alternative", "inconsistent",
            "timeout", "delay", "lagging",
        ],
        "Low": [
            "minor", "question", "how to", "request", "suggestion",
            "enhancement", "nice to have", "cosmetic", "when possible",
            "information", "inquiry", "scheduled", "planned", "feedback",
            "training", "documentation", "tutorial",
        ],
    }

    # ═══════════════════════════════════════════════════════════════════
    #  Category-Specific Bigram Dictionaries  (NEW in v4)
    # ═══════════════════════════════════════════════════════════════════

    CATEGORY_BIGRAMS = {
        "Network": [
            "slow internet", "no internet", "wifi disconnecting", "vpn connection",
            "network drive", "lan cable", "packet loss", "high latency",
            "dns resolution", "ip address", "access point", "network switch",
            "proxy server", "ip conflict", "dhcp not", "ethernet port",
            "firewall blocking", "gateway unreachable", "bandwidth low",
            "connection drops", "network connectivity", "wireless network",
        ],
        "Software": [
            "blue screen", "not launching", "cannot login", "error code",
            "license expired", "not responding", "software installation",
            "application crashes", "update failed", "password reset",
            "office activation", "browser not", "boot failure",
            "configuration error", "driver update", "plugin not",
            "not compatible", "system instability", "keeps crashing",
            "runtime error", "dll missing", "registry error",
        ],
        "Hardware": [
            "loud noise", "screen flickering", "not powering", "high cpu",
            "battery draining", "usb ports", "docking station", "dead pixels",
            "not charging", "disk clicking", "fan noise", "overheating shutting",
            "power supply", "graphics card", "health warning", "display not",
            "not detected", "startup hang", "low memory", "slow performance",
        ],
        "Printer": [
            "paper jam", "print queue", "toner cartridge", "blank pages",
            "offline printer", "print spooler", "duplex printing",
            "ink levels", "paper tray", "scan email", "printer driver",
            "faded output", "head clogged", "colour printing", "print job",
            "network printer", "paper feed", "roller replacement",
        ],
    }

    # ═══════════════════════════════════════════════════════════════════
    #  Urgency, Negation, Entity, and NEW Feature Patterns
    # ═══════════════════════════════════════════════════════════════════

    URGENCY_WORDS = [
        "urgent", "critical", "immediately", "asap", "emergency",
        "down", "outage", "blocking", "stopped", "crashed",
        "failure", "broken", "dead", "unresponsive", "cannot",
    ]

    NEGATION_PATTERNS = [
        r"\bnot\s+working\b", r"\bcannot\b", r"\bcan't\b", r"\bunable\b",
        r"\bwon't\b", r"\bdoesn't\b", r"\bdoes\s+not\b", r"\bfailed\b",
        r"\bfailing\b", r"\bnot\s+responding\b", r"\bnot\s+connecting\b",
        r"\bnot\s+loading\b", r"\bnot\s+opening\b", r"\bno\s+access\b",
        r"\bno\s+connection\b", r"\bno\s+internet\b",
    ]

    ENTITY_PATTERNS = [
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",       # IP addresses
        r"\bfloor\s+\d+\b",                                 # Floor references
        r"\b(?:pc|sw|rt|srv|ws)-[\w-]+\b",                  # Device IDs
        r"\b(?:port|ext)\s*:?\s*\d+\b",                     # Port/extension
        r"\b\d+\s*(?:users?|workstations?|systems?)\b",     # Affected count
    ]

    # v4: Temporal urgency patterns
    TEMPORAL_PATTERNS = [
        r"\bsince\s+(?:morning|yesterday|last\s+\w+)\b",
        r"\bfor\s+\d+\s*(?:hours?|minutes?|days?)\b",
        r"\bright\s+now\b", r"\bpast\s+\d+\b",
        r"\bsince\s+\d+\b", r"\ball\s+day\b", r"\ball\s+morning\b",
        r"\bwhole\s+(?:day|week|morning)\b",
        r"\bkeeps?\s+(?:happening|recurring|crashing|disconnecting)\b",
    ]

    # v4: Impact scope patterns
    IMPACT_PATTERNS = [
        r"\bentire\s+(?:floor|department|building|office|team)\b",
        r"\ball\s+(?:users|employees|staff|workstations|systems)\b",
        r"\bwhole\s+(?:floor|department|office|team|network)\b",
        r"\beveryone\b", r"\bmultiple\s+(?:users|systems|departments)\b",
        r"\b\d{2,}\s+(?:users?|people|employees?|workstations?)\b",
    ]

    # v4: Technical jargon wordset
    TECH_JARGON = {
        "bios", "firmware", "kernel", "daemon", "tcp", "udp", "http",
        "https", "ssl", "tls", "ssh", "ftp", "smtp", "imap", "pop3",
        "dns", "dhcp", "nat", "vlan", "vpn", "ipsec", "ldap", "ntfs",
        "fat32", "raid", "ssd", "nvme", "sata", "gpu", "cpu", "ram",
        "rom", "bsod", "dll", "exe", "msi", "iso", "api", "url",
        "xml", "json", "csv", "sql", "registry", "driver", "spooler",
        "subnet", "gateway", "router", "switch", "firewall", "proxy",
        "cache", "buffer", "stack", "heap", "thread", "process",
        "latency", "bandwidth", "throughput", "packet", "payload",
    }

    # v4: Action/verb-like words for POS heuristic
    ACTION_WORDS = {
        "restart", "reboot", "install", "uninstall", "update", "upgrade",
        "configure", "connect", "disconnect", "reset", "repair", "replace",
        "download", "upload", "delete", "remove", "add", "fix", "resolve",
        "troubleshoot", "diagnose", "scan", "backup", "restore", "deploy",
        "migrate", "patch", "format", "partition", "enable", "disable",
        "login", "logout", "authenticate", "authorize", "assign", "check",
    }

    # v4: Negative sentiment words for priority
    NEGATIVE_WORDS = {
        "frustrated", "angry", "annoyed", "terrible", "awful", "horrible",
        "unacceptable", "ridiculous", "pathetic", "useless", "waste",
        "broken", "ruined", "destroyed", "impossible", "nightmare",
        "painful", "suffering", "struggling", "desperate", "stuck",
        "failing", "failed", "worst", "never", "nothing", "nowhere",
    }

    # v4: Repeated emphasis patterns
    EMPHASIS_PATTERNS = [
        r"\b(\w+)\s+\1\b",                                  # repeated word
        r"\bvery\s+very\b", r"\bso\s+so\b",
        r"\bkeeps?\s+\w+ing\b",                              # keeps crashing
        r"\bagain\s+and\s+again\b", r"\bover\s+and\s+over\b",
        r"\bstill\s+not\b", r"\bstill\s+(?:broken|down|failing)\b",
        r"!{2,}",                                            # multiple !!
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

    # ═══════════════════════════════════════════════════════════════════
    #  Initialisation & Training
    # ═══════════════════════════════════════════════════════════════════

    # Feature name maps for explainability
    CATEGORY_FEATURE_NAMES = [
        "text_length", "word_count",
        "kw_network", "kw_software", "kw_hardware", "kw_printer",
        "entity_density",
        "sentence_count", "avg_word_length", "special_char_density",
        "has_question_mark",
        "bigram_network", "bigram_software", "bigram_hardware", "bigram_printer",
        "tech_jargon_density", "action_word_ratio",
    ]

    PRIORITY_FEATURE_NAMES = [
        "text_length", "urgency_count", "negation_count",
        "entity_density", "exclamation_count", "uppercase_ratio",
        "affected_user_count",
        "temporal_urgency", "impact_scope", "emphasis_count",
        "negative_sentiment", "has_error_code", "word_diversity_ttr",
    ]

    def __init__(self):
        """Initialize and train the classifier on synthetic data."""
        self.category_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=8000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.priority_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words="english",
            sublinear_tf=True,
        )

        self.category_model = LogisticRegression(
            max_iter=2000,
            C=1.5,
            solver="lbfgs",
            class_weight="balanced",
        )
        self.priority_model = LogisticRegression(
            max_iter=2000,
            C=1.5,
            solver="lbfgs",
            class_weight="balanced",
        )

        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()

        self._is_trained = False
        self._num_cat_features = len(self.CATEGORY_FEATURE_NAMES)
        self._num_pri_features = len(self.PRIORITY_FEATURE_NAMES)

        self._train_on_synthetic_data()

    # ═══════════════════════════════════════════════════════════════════
    #  Synthetic Training Corpus  (expanded for v4 — ~1500 samples)
    # ═══════════════════════════════════════════════════════════════════

    def _generate_training_corpus(self):
        """Generate a large synthetic IT support corpus from templates."""
        corpus = []

        # ── Network tickets ──
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
            "Routing loop detected between floor {floor} and {floor} switches",
            "NTP sync failing across all network devices",
        ]

        # ── Software tickets ──
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
            "VBS script error on user login to domain",
            "PowerShell execution policy blocking automation script",
            "Windows Defender quarantined a critical DLL file",
            "Application shows access denied error code {code}",
        ]

        # ── Hardware tickets ──
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
            "Laptop keyboard backlight stopped working",
        ]

        # ── Printer tickets ──
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
            "Print jobs disappear from queue without printing",
        ]

        # ── Other / general tickets ──
        other_templates = [
            "How to set up email on mobile phone",
            "Request for new employee account creation",
            "Need access card for server room",
            "Suggestion for improving IT portal interface",
            "Question about data backup procedures",
            "Information needed about software procurement",
            "Request for ergonomic assessment of desk setup",
            "A/C is not working in the office area",
            "Air conditioner is not cooling in our cabin",
            "Conference room cooling issue needs facilities support",
            "Room temperature is too hot because the AC stopped working",
            "Need facilities team support for air conditioning problem",
            "How to join the video conference bridge",
            "Need training on new HR management system",
            "General inquiry about IT policies",
            "How do I change my email signature",
            "Request to add shared mailbox access",
            "Need guidance on file sharing best practices",
            "When is the next scheduled maintenance window",
            "Feedback on the new ticketing system interface",
        ]

        # Generate variations
        import random
        random.seed(42)

        # Network: 35 templates × 5 variations = 175
        for template in network_templates:
            for _ in range(5):
                text = template.format(
                    floor=random.randint(1, 10),
                    desk=random.randint(100, 900),
                    n=random.randint(3, 30),
                )
                corpus.append(
                    (text, "Network", random.choice(["High", "Medium", "Medium", "Medium"]))
                )

        # Software: 35 templates × 5 variations = 175
        for template in software_templates:
            for _ in range(5):
                text = template.format(
                    code=random.randint(1000, 9999),
                    n=random.randint(10, 95),
                )
                corpus.append(
                    (text, "Software", random.choice(["Medium", "Medium", "Low", "Medium"]))
                )

        # Hardware: 35 templates × 5 variations = 175
        for template in hardware_templates:
            for _ in range(5):
                text = template.format(n=random.randint(10, 120))
                corpus.append(
                    (text, "Hardware", random.choice(["Medium", "Low", "Medium", "Medium"]))
                )

        # Printer: 25 templates × 5 variations = 125
        for template in printer_templates:
            for _ in range(5):
                text = template.format(
                    floor=random.randint(1, 8),
                    n=random.randint(5, 60),
                    code=random.randint(100, 999),
                )
                corpus.append(
                    (text, "Printer", random.choice(["Low", "Medium", "Low", "Low"]))
                )

        # Other: 15 templates × 5 variations = 75
        for template in other_templates:
            for _ in range(5):
                corpus.append((template, "Other", "Low"))

        # ── High-priority variants (with urgency prefixes) ──
        high_pri_prefixes = [
            "URGENT: ", "CRITICAL: ", "All users affected — ",
            "Production system down — ", "Emergency: ", "Cannot work — ",
            "Entire floor affected — ", "ASAP needed — ",
            "IMMEDIATELY REQUIRED: ", "BLOCKING ALL WORK: ",
            "Since morning no fix — ", "Escalate now — ",
        ]
        for prefix in high_pri_prefixes:
            for template in network_templates[:8]:
                text = prefix + template.format(floor=random.randint(1, 10), desk=100, n=5)
                corpus.append((text, "Network", "High"))
            for template in software_templates[:6]:
                text = prefix + template.format(code=random.randint(1000, 9999), n=50)
                corpus.append((text, "Software", "High"))
            for template in hardware_templates[:5]:
                text = prefix + template.format(n=random.randint(5, 30))
                corpus.append((text, "Hardware", "High"))
            for template in printer_templates[:3]:
                text = prefix + template.format(floor=random.randint(1, 6), n=20, code=500)
                corpus.append((text, "Printer", "High"))

        # ── Medium-priority with context phrases ──
        medium_prefixes = [
            "Intermittent issue: ", "Sometimes: ",
            "Workaround available but: ", "Partial outage: ",
            "Slow but functional: ", "Affecting a few users: ",
        ]
        for prefix in medium_prefixes:
            for template in network_templates[:5]:
                text = prefix + template.format(floor=random.randint(1, 8), desk=200, n=10)
                corpus.append((text, "Network", "Medium"))
            for template in software_templates[:5]:
                text = prefix + template.format(code=random.randint(1000, 9999), n=40)
                corpus.append((text, "Software", "Medium"))

        # ── Low-priority with question/request framing ──
        low_prefixes = [
            "When possible: ", "Minor issue: ", "Just a question: ",
            "No rush but: ", "Nice to have: ", "For future reference: ",
        ]
        for prefix in low_prefixes:
            for template in other_templates[:5]:
                corpus.append((prefix + template, "Other", "Low"))
            for template in software_templates[:3]:
                text = prefix + template.format(code=1234, n=50)
                corpus.append((text, "Software", "Low"))

        # ── Frustrated / angry tone tickets (High priority signals) ──
        frustrated_templates = [
            "I am so frustrated! {base} This is happening again and again!",
            "Unacceptable! {base} We have been struggling with this all day!",
            "This is ridiculous — {base} Nothing works!!",
            "I cannot work at all! {base} Need immediate help!",
            "Very very slow {base} — desperate for a fix ASAP",
        ]
        base_issues = [
            "Internet is down on floor 3.",
            "Outlook keeps crashing every 5 minutes.",
            "Laptop overheating and shutting down.",
            "Printer has been offline for 2 days now.",
        ]
        for ft in frustrated_templates:
            for base in base_issues:
                corpus.append((ft.format(base=base), "Network" if "internet" in base.lower() else
                              "Software" if "outlook" in base.lower() else
                              "Hardware" if "laptop" in base.lower() else "Printer", "High"))

        return corpus

    def _train_on_synthetic_data(self):
        """Train both category and priority models on synthetic corpus."""
        corpus = self._generate_training_corpus()

        texts = [item[0] for item in corpus]
        categories = [item[1] for item in corpus]
        priorities = [item[2] for item in corpus]

        # ── Train Category Model ──
        cat_labels = self.category_encoder.fit_transform(categories)
        X_cat = self.category_vectorizer.fit_transform(texts)

        extra_cat = np.array([self._extract_features(t) for t in texts])
        X_cat_full = np.hstack([X_cat.toarray(), extra_cat])

        self.category_model.fit(X_cat_full, cat_labels)

        # ── Train Priority Model ──
        pri_labels = self.priority_encoder.fit_transform(priorities)
        X_pri = self.priority_vectorizer.fit_transform(texts)

        extra_pri = np.array([self._extract_priority_features(t) for t in texts])
        X_pri_full = np.hstack([X_pri.toarray(), extra_pri])

        self.priority_model.fit(X_pri_full, pri_labels)

        self._is_trained = True
        self._corpus_size = len(corpus)

    # ═══════════════════════════════════════════════════════════════════
    #  Feature Engineering — Category  (17 features total)
    # ═══════════════════════════════════════════════════════════════════

    def _extract_features(self, text):
        """
        Extract engineered features for category prediction.
        Returns a list of 17 normalised features.
        """
        text_lower = text.lower()
        words = text_lower.split()
        features = []

        # 1. Text length (normalised)
        features.append(min(len(text) / 500.0, 1.0))

        # 2. Word count (normalised)
        features.append(min(len(words) / 50.0, 1.0))

        # 3-6. Keyword match density per category
        for cat in ["Network", "Software", "Hardware", "Printer"]:
            keywords = self.CATEGORY_KEYWORDS[cat]
            count = sum(1 for kw in keywords if kw in text_lower)
            features.append(count / max(len(keywords), 1))

        # 7. IT entity density
        entity_count = sum(
            len(re.findall(pat, text_lower, re.IGNORECASE))
            for pat in self.ENTITY_PATTERNS
        )
        features.append(min(entity_count / 5.0, 1.0))

        # ── NEW v4 Features ──────────────────────────────────────────

        # 8. Sentence count (normalised)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features.append(min(len(sentences) / 5.0, 1.0))

        # 9. Average word length
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            features.append(min(avg_len / 10.0, 1.0))
        else:
            features.append(0.0)

        # 10. Special character density (non-alphanumeric, non-space)
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features.append(min(special / max(len(text), 1), 1.0))

        # 11. Question mark presence
        features.append(1.0 if "?" in text else 0.0)

        # 12-15. Category-specific bigram hit count
        for cat in ["Network", "Software", "Hardware", "Printer"]:
            bigrams = self.CATEGORY_BIGRAMS.get(cat, [])
            hit_count = sum(1 for bg in bigrams if bg in text_lower)
            features.append(min(hit_count / max(len(bigrams), 1) * 3, 1.0))

        # 16. Technical jargon density
        word_tokens = re.findall(r'\b\w+\b', text_lower)
        jargon_count = sum(1 for w in word_tokens if w in self.TECH_JARGON)
        features.append(min(jargon_count / max(len(word_tokens), 1) * 5, 1.0))

        # 17. Action/verb-like word ratio
        action_count = sum(1 for w in word_tokens if w in self.ACTION_WORDS)
        features.append(min(action_count / max(len(word_tokens), 1) * 5, 1.0))

        return features

    # ═══════════════════════════════════════════════════════════════════
    #  Feature Engineering — Priority  (13 features total)
    # ═══════════════════════════════════════════════════════════════════

    def _extract_priority_features(self, text):
        """
        Extract engineered features for priority prediction.
        Returns a list of 13 normalised features.
        """
        text_lower = text.lower()
        word_tokens = re.findall(r'\b\w+\b', text_lower)
        features = []

        # 1. Text length
        features.append(min(len(text) / 500.0, 1.0))

        # 2. Urgency word count
        urgency_count = sum(1 for w in self.URGENCY_WORDS if w in text_lower)
        features.append(min(urgency_count / 5.0, 1.0))

        # 3. Negation pattern count
        negation_count = sum(
            len(re.findall(pat, text_lower))
            for pat in self.NEGATION_PATTERNS
        )
        features.append(min(negation_count / 3.0, 1.0))

        # 4. IT entity density
        entity_count = sum(
            len(re.findall(pat, text_lower, re.IGNORECASE))
            for pat in self.ENTITY_PATTERNS
        )
        features.append(min(entity_count / 5.0, 1.0))

        # 5. Exclamation marks (urgency signal)
        features.append(min(text.count("!") / 3.0, 1.0))

        # 6. Uppercase ratio (shouting = urgency)
        if len(text) > 0:
            upper_chars = sum(1 for c in text if c.isupper())
            features.append(min(upper_chars / len(text), 1.0))
        else:
            features.append(0.0)

        # 7. Affected user count detection
        affected_match = re.search(
            r"(\d+)\s*(?:users?|workstations?|systems?|people|employees?)",
            text_lower,
        )
        if affected_match:
            count = int(affected_match.group(1))
            features.append(min(count / 50.0, 1.0))
        else:
            features.append(0.0)

        # ── NEW v4 Features ──────────────────────────────────────────

        # 8. Temporal urgency (time-related phrases)
        temporal_hits = sum(
            len(re.findall(pat, text_lower))
            for pat in self.TEMPORAL_PATTERNS
        )
        features.append(min(temporal_hits / 3.0, 1.0))

        # 9. Impact scope (how many are affected)
        impact_hits = sum(
            len(re.findall(pat, text_lower))
            for pat in self.IMPACT_PATTERNS
        )
        features.append(min(impact_hits / 2.0, 1.0))

        # 10. Repeated emphasis / frustration
        emphasis_hits = sum(
            len(re.findall(pat, text_lower))
            for pat in self.EMPHASIS_PATTERNS
        )
        features.append(min(emphasis_hits / 3.0, 1.0))

        # 11. Negative sentiment density
        neg_count = sum(1 for w in word_tokens if w in self.NEGATIVE_WORDS)
        features.append(min(neg_count / 3.0, 1.0))

        # 12. Has error code (presence of numeric error code patterns)
        has_error = 1.0 if re.search(
            r"\b(?:error|err|code|fault)\s*:?\s*(?:0x)?[\da-fA-F]{3,}\b", text_lower
        ) else 0.0
        features.append(has_error)

        # 13. Word diversity — Type-Token Ratio
        if word_tokens:
            ttr = len(set(word_tokens)) / len(word_tokens)
            features.append(ttr)
        else:
            features.append(0.0)

        return features

    # ═══════════════════════════════════════════════════════════════════
    #  Keyword Boosting
    # ═══════════════════════════════════════════════════════════════════

    def _keyword_boost_category(self, text):
        """Return keyword-based scores for each category."""
        text_lower = text.lower()
        scores = {}
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            matched = sum(1 for kw in keywords if kw in text_lower)
            scores[cat] = matched

        # Also boost from bigrams
        for cat, bigrams in self.CATEGORY_BIGRAMS.items():
            bigram_hits = sum(1 for bg in bigrams if bg in text_lower)
            scores[cat] = scores.get(cat, 0) + bigram_hits * 2  # bigrams worth 2x

        return scores

    def _keyword_boost_priority(self, text):
        """Return keyword-based priority signal."""
        text_lower = text.lower()
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            matched = sum(1 for kw in keywords if kw in text_lower)
            if matched >= 1:
                return priority, matched
        return "Medium", 0

    def _is_facilities_issue(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.FACILITY_PATTERNS)

    # ═══════════════════════════════════════════════════════════════════
    #  Confidence Calibration  (NEW in v4)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _calibrate_confidence(raw_confidence, keyword_strength, feature_agreement):
        """
        Post-hoc calibration of ML confidence using keyword and feature signals.
        Prevents over-confident or under-confident scores.
        """
        # Start with raw ML confidence
        cal = raw_confidence

        # Boost if keywords strongly agree
        if keyword_strength >= 3:
            cal = min(cal + 0.10, 0.98)
        elif keyword_strength >= 2:
            cal = min(cal + 0.05, 0.95)

        # Penalise if very low feature agreement
        if feature_agreement < 0.2:
            cal = max(cal - 0.10, 0.15)

        # Dampen extreme confidence on edge cases
        if cal > 0.95 and keyword_strength < 2:
            cal = 0.90

        return round(cal, 3)

    # ═══════════════════════════════════════════════════════════════════
    #  Explainability — Top Contributing Features  (NEW in v4)
    # ═══════════════════════════════════════════════════════════════════

    def _explain_prediction(self, features, feature_names, prediction_type="category"):
        """
        Return top contributing features for the prediction.
        """
        feature_contributions = []
        for i, (name, value) in enumerate(zip(feature_names, features)):
            if value > 0.01:  # only non-trivial features
                feature_contributions.append({
                    "feature": name,
                    "value": round(float(value), 4),
                    "impact": "high" if value > 0.5 else "medium" if value > 0.2 else "low",
                })

        # Sort by value descending, take top 5
        feature_contributions.sort(key=lambda x: x["value"], reverse=True)
        return feature_contributions[:5]

    # ═══════════════════════════════════════════════════════════════════
    #  Main Classification
    # ═══════════════════════════════════════════════════════════════════

    def classify(self, text):
        """
        Classify a ticket description and return full result including
        confidences, method, and explainability data.

        Returns: dict with keys:
            category, priority, confidence,
            category_confidence, priority_confidence,
            ml_category, ml_priority, method,
            top_features, feature_breakdown
        """
        if not self._is_trained:
            return self._fallback_classify(text)

        facilities_issue = self._is_facilities_issue(text)

        # ── ML Category Prediction ──
        X_cat = self.category_vectorizer.transform([text])
        cat_features = self._extract_features(text)
        extra_cat = np.array([cat_features])
        X_cat_full = np.hstack([X_cat.toarray(), extra_cat])

        cat_proba = self.category_model.predict_proba(X_cat_full)[0]
        ml_cat_idx = np.argmax(cat_proba)
        ml_category = self.category_encoder.inverse_transform([ml_cat_idx])[0]
        ml_cat_confidence = float(cat_proba[ml_cat_idx])

        # ── ML Priority Prediction ──
        X_pri = self.priority_vectorizer.transform([text])
        pri_features = self._extract_priority_features(text)
        extra_pri = np.array([pri_features])
        X_pri_full = np.hstack([X_pri.toarray(), extra_pri])

        pri_proba = self.priority_model.predict_proba(X_pri_full)[0]
        ml_pri_idx = np.argmax(pri_proba)
        ml_priority = self.priority_encoder.inverse_transform([ml_pri_idx])[0]
        ml_pri_confidence = float(pri_proba[ml_pri_idx])

        # ── Keyword Boosting ──
        kw_scores = self._keyword_boost_category(text)
        kw_priority, kw_pri_count = self._keyword_boost_priority(text)

        # Hybrid: if keyword match is strong, override low-confidence ML
        category = ml_category
        category_confidence = ml_cat_confidence

        best_kw_cat = max(kw_scores, key=kw_scores.get)
        if kw_scores[best_kw_cat] >= 2 and ml_cat_confidence < 0.6:
            category = best_kw_cat
            category_confidence = min(0.85, ml_cat_confidence + kw_scores[best_kw_cat] * 0.1)
        elif kw_scores[best_kw_cat] >= 3:
            if best_kw_cat == ml_category:
                category_confidence = min(0.98, ml_cat_confidence + 0.15)
            else:
                category = best_kw_cat
                category_confidence = min(0.80, kw_scores[best_kw_cat] * 0.15)

        # Handle "Other"
        if category_confidence < 0.3 and max(kw_scores.values()) == 0:
            category = "Other"
            category_confidence = 0.5
        elif facilities_issue:
            category = "Other"
            category_confidence = max(category_confidence, 0.92)

        # Priority hybrid
        priority = ml_priority
        priority_confidence = ml_pri_confidence

        if kw_pri_count >= 2 and ml_pri_confidence < 0.6:
            priority = kw_priority
            priority_confidence = min(0.85, ml_pri_confidence + kw_pri_count * 0.1)

        # Negation/urgency override
        text_lower = text.lower()
        negation_count = sum(
            len(re.findall(p, text_lower)) for p in self.NEGATION_PATTERNS
        )
        if negation_count >= 2 and priority != "High":
            priority = "High"
            priority_confidence = max(priority_confidence, 0.7)

        # v4: Impact scope override — if large scope detected, boost to High
        impact_hits = sum(
            len(re.findall(pat, text_lower)) for pat in self.IMPACT_PATTERNS
        )
        if impact_hits >= 2 and priority != "High":
            priority = "High"
            priority_confidence = max(priority_confidence, 0.75)

        # v4: Temporal urgency boost
        temporal_hits = sum(
            len(re.findall(pat, text_lower)) for pat in self.TEMPORAL_PATTERNS
        )
        if temporal_hits >= 2 and priority == "Low":
            priority = "Medium"
            priority_confidence = max(priority_confidence, 0.65)

        # v4: Confidence calibration
        kw_strength = kw_scores.get(category, 0)
        cat_feature_max = max(cat_features[2:6]) if cat_features[2:6] else 0  # keyword density features
        category_confidence = self._calibrate_confidence(
            category_confidence, kw_strength, cat_feature_max
        )
        priority_confidence = self._calibrate_confidence(
            priority_confidence, kw_pri_count,
            max(pri_features[1:3]) if pri_features[1:3] else 0  # urgency + negation features
        )

        # Overall confidence
        overall_confidence = (category_confidence + priority_confidence) / 2

        # v4: Explainability
        cat_explanation = self._explain_prediction(
            cat_features, self.CATEGORY_FEATURE_NAMES, "category"
        )
        pri_explanation = self._explain_prediction(
            pri_features, self.PRIORITY_FEATURE_NAMES, "priority"
        )

        return {
            "category":             category,
            "priority":             priority,
            "confidence":           round(overall_confidence, 3),
            "category_confidence":  round(category_confidence, 3),
            "priority_confidence":  round(priority_confidence, 3),
            "ml_category":          ml_category,
            "ml_priority":          ml_priority,
            "method":               "hybrid_ml_keyword_v4",
            "top_features": {
                "category": cat_explanation,
                "priority": pri_explanation,
            },
            "feature_breakdown": {
                "category_features": {
                    name: round(float(val), 4)
                    for name, val in zip(self.CATEGORY_FEATURE_NAMES, cat_features)
                    if val > 0.001
                },
                "priority_features": {
                    name: round(float(val), 4)
                    for name, val in zip(self.PRIORITY_FEATURE_NAMES, pri_features)
                    if val > 0.001
                },
                "keyword_scores": {
                    k: v for k, v in kw_scores.items() if v > 0
                },
            },
        }

    def _fallback_classify(self, text):
        """Pure keyword fallback if ML model fails to train."""
        text_lower = text.lower()
        if self._is_facilities_issue(text):
            return {
                "category":             "Other",
                "priority":             "Medium",
                "confidence":           0.9,
                "category_confidence":  0.9,
                "priority_confidence":  0.5,
                "ml_category":          "Other",
                "ml_priority":          "Medium",
                "method":               "keyword_fallback",
                "top_features":         {"category": [], "priority": []},
                "feature_breakdown":    {},
            }

        cat_scores = {cat: 0 for cat in self.CATEGORY_KEYWORDS}
        for cat, kws in self.CATEGORY_KEYWORDS.items():
            for kw in kws:
                if kw in text_lower:
                    cat_scores[cat] += 1

        category = max(cat_scores, key=cat_scores.get)
        if cat_scores[category] == 0:
            category = "Other"

        priority = "Medium"
        for pri, kws in self.PRIORITY_KEYWORDS.items():
            if any(kw in text_lower for kw in kws):
                priority = pri
                break

        return {
            "category":             category,
            "priority":             priority,
            "confidence":           min(1.0, cat_scores.get(category, 0) / 3),
            "category_confidence":  min(1.0, cat_scores.get(category, 0) / 3),
            "priority_confidence":  0.5,
            "ml_category":          category,
            "ml_priority":          priority,
            "method":               "keyword_fallback",
            "top_features":         {"category": [], "priority": []},
            "feature_breakdown":    {},
        }

    # ═══════════════════════════════════════════════════════════════════
    #  Model Metadata (for health/debug endpoints)
    # ═══════════════════════════════════════════════════════════════════

    def get_model_info(self):
        """Return model metadata for system introspection."""
        return {
            "version":                "4.0.0",
            "method":                 "TF-IDF + Logistic Regression (Hybrid + Keyword Boost)",
            "is_trained":             self._is_trained,
            "corpus_size":            getattr(self, "_corpus_size", 0),
            "category_features":      self._num_cat_features,
            "priority_features":      self._num_pri_features,
            "tfidf_cat_vocab_size":   len(self.category_vectorizer.vocabulary_) if self._is_trained else 0,
            "tfidf_pri_vocab_size":   len(self.priority_vectorizer.vocabulary_) if self._is_trained else 0,
            "categories":             list(self.category_encoder.classes_) if self._is_trained else [],
            "priorities":             list(self.priority_encoder.classes_) if self._is_trained else [],
            "engineered_features": {
                "category": self.CATEGORY_FEATURE_NAMES,
                "priority": self.PRIORITY_FEATURE_NAMES,
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    #  TF-IDF Vector for Duplicate Detection
    # ═══════════════════════════════════════════════════════════════════

    def get_tfidf_vector(self, text):
        """
        Generate a simple term-frequency vector for cosine similarity
        duplicate detection. Uses word-level frequencies.
        """
        words = re.findall(r'\w+', text.lower())
        freq = {}
        for w in words:
            if len(w) > 2:
                freq[w] = freq.get(w, 0) + 1
        total = sum(freq.values()) or 1
        return {k: round(v / total, 4) for k, v in freq.items()}

    def cosine_similarity(self, vec_a, vec_b):
        """Compute cosine similarity between two sparse dicts."""
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        return dot / (mag_a * mag_b) if mag_a * mag_b else 0.0


# ── Module-level singleton ──────────────────────────────────────────────
classifier = EnhancedTicketClassifier()
