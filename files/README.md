# D Desk AI — Smart Helpdesk Ticketing Solution
## POWERGRID IT Services | v4.0.0

---

## 🗂️ Project Structure

```
d-desk-ai/
├── index.html          ← Landing page (role-based login)
├── employee.html       ← Employee chat portal
├── admin.html          ← Admin dashboard
├── technician.html     ← Technician portal with retro terminal
├── run.py              ← Application entry point
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
└── app/
    ├── __init__.py     ← FastAPI application factory
    ├── config.py       ← Centralized configuration
    ├── ml/             ← Pretrained ML models & classifiers
    ├── routes/         ← API route handlers (microservices)
    └── services/       ← Business logic & database layer
```

---

## 🚀 Quick Start

### 1. Backend Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Firebase credentials, Groq API key, and demo passwords
python run.py
# → Server runs at http://localhost:8000
# → API docs at http://localhost:8000/docs
```

### 2. Demo Accounts

Demo user accounts are seeded on first startup. Configure credentials via environment variables in `.env`:

| Role       | Username | Password Source |
|------------|----------|-----------------|
| Admin      | *(set via `DEMO_ADMIN_EMAIL`)* | `DEMO_ADMIN_PASSWORD` |
| Admin      | admin    | `DEMO_DEFAULT_PASSWORD` |
| Employee   | emp001   | `DEMO_DEFAULT_PASSWORD` |
| Technician | tech01   | `DEMO_DEFAULT_PASSWORD` |

See `.env.example` for all required variables.

### 3. Frontend

Open `http://localhost:8000` in any browser — the FastAPI server serves all frontend pages.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND (HTML/CSS/JS)                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ Employee │  │  Admin   │  │      Technician        │  │
│  │  Portal  │  │Dashboard │  │  Portal + Retro Term   │  │
│  └────┬─────┘  └────┬─────┘  └──────────┬────────────┘  │
└───────┼─────────────┼───────────────────┼───────────────┘
        │             │                   │
        ▼             ▼                   ▼
┌──────────────────────────────────────────────────────────┐
│                   FastAPI Backend                         │
│  /api/tickets    /api/chat    /api/insights               │
│  /api/circulars  /api/stats   /api/analytics              │
│                                                           │
│  ┌───────────────────┐  ┌───────────────────────────┐    │
│  │  ML Classifier     │  │  Groq LLM (LLaMA 3.1)     │    │
│  │  Logistic Reg +    │  │  - Chat responses          │    │
│  │  TF-IDF            │  │  - Auto-fix suggestions    │    │
│  │  scikit-learn      │  │  - Admin insights          │    │
│  │  Categories:       │  │  - Report generation       │    │
│  │  Network/Software  │  └───────────────────────────┘    │
│  │  Hardware/Printer  │                                   │
│  │  /Other            │  ┌───────────────────────────┐    │
│  └───────────────────┘  │  Duplicate Detection       │    │
│                          │  Cosine Similarity         │    │
│  ┌───────────────────┐   │  (TF-IDF vectors)          │    │
│  │  Cloud Firestore   │   └───────────────────────────┘    │
│  │  (Firebase Admin)  │                                   │
│  │  users             │                                   │
│  │  tickets           │                                   │
│  │  circulars         │                                   │
│  └───────────────────┘                                   │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **AI Auto-Fix Before Ticket** | LLM suggests instant fixes; employee marks resolved → no ticket created |
| **Duplicate Detection** | Cosine similarity on TF-IDF vectors blocks redundant tickets |
| **ML Classification** | Logistic Regression classifies: Network / Software / Hardware / Printer / Other |
| **Priority Prediction** | Keyword-based priority (High / Medium / Low) |
| **Admin AI Insights** | Groq/LLaMA generates trend analysis, recommendations, and reports |
| **Circular Management** | Admin broadcasts notices → Technicians see them in retro terminal |
| **Retro Terminal** | Green-on-black CRT-style interface for technician communications |
| **Multi-Source Ingestion** | Accepts tickets from chat, email, GLPI, Solman via unified API |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/login` | Authenticate user |
| GET | `/api/stats` | Dashboard statistics |
| POST | `/api/tickets` | Create ticket (full AI pipeline) |
| GET | `/api/tickets` | List tickets (filterable) |
| PATCH | `/api/tickets/{id}` | Update ticket status |
| POST | `/api/chat` | LLM-powered chat response |
| GET | `/api/insights` | AI-generated admin insights |
| POST | `/api/circulars` | Send admin circular |
| GET | `/api/circulars` | List circulars |
| GET | `/api/analytics/category-distribution` | Category stats |
| GET | `/api/analytics/daily-volume` | Daily ticket volume |
| GET | `/api/health` | Health check |

---

## 🎨 Design Theme — D Desk AI

- **Color palette**: Deep navy background `#04060f` + Neon blue `#00d4ff` + Neon green `#00ff88`
- **Typography**: Orbitron (headings) + Exo 2 (body) + Share Tech Mono (terminal/data)
- **Effects**: Glassmorphism cards, neural network canvas animation, CRT scanlines
- **Technician terminal**: Retro green-on-black CRT monitor with typewriter animation and interactive commands
- **Micro-interactions**: Hover glow effects, typing indicators, smooth transitions

---

## 🔧 Production Checklist

- [ ] Use strong, unique passwords for all accounts
- [ ] Configure `GROQ_API_KEY` environment variable
- [ ] Deploy frontend to nginx / serve via FastAPI static
- [ ] Enable HTTPS
- [ ] Set up email ingestion (IMAP parser for unified ticket source)
- [ ] Connect GLPI / Solman via their REST APIs
