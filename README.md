# 🏛️ Tamil Nadu Welfare Schemes AI Portal

An enterprise-grade, citizen-centric RAG (Retrieval-Augmented Generation) assistant that parses, vectorizes, and serves verified insights from official Tamil Nadu Government policy directories. 

This project was built to solve the difficulty citizens face when navigating complex tabular data and nested criteria on government portals.

---

## 🚀 Key Features & Hackathon Highlights

* **Tabular Context Retention:** Built with a custom `BeautifulSoup` parsing pipeline that reads web data row-by-row rather than as flat text blocks. This guarantees that subsidy titles never get separated from their corresponding monetary benefits or eligibility rules inside the vector database.
* **Production-Grade Separated Architecture:** Divided into two isolated cycles: an **Offline Ingestion Pipeline** and an **Online Inference Web App**. The frontend never triggers live web scrapes during chat sessions, ensuring instant UI rendering and zero lag.
* **Deterministic Guardrails:** Configured with a system prompt and a temperature of `0.0` to eliminate LLM hallucinations. If an engineering program or welfare asset is missing from the local matrix, the engine safely falls back without inventing facts.
* **Enterprise Dashboard UI:** Features a custom CSS-overhauled side navigation bar, modern dark contrast styling, and responsive status tracking designed to resemble a high-end state deployment.

---

## ⚙️ Tech Stack Architecture

* **Frontend Dashboard:** [Streamlit](https://streamlit.io/) (with custom injected CSS web layouts)
* **Orchestration Framework:** [LangChain](https://www.langchain.com/) (LCEL - LangChain Expression Language)
* **Embedding Model:** Local HuggingFace Transformers (`all-MiniLM-L6-v2`)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (Local persistent storage core)
* **Foundation LLM:** OpenAI (`gpt-4o-mini`)

---

## 🛠️ Project Structure

```text
tn_website/
│
├── chroma_db/          # Persistent local vector storage directory (Auto-generated)
├── venv/               # Python local virtual environment
│
├── ingest.py           # Offline Data Scraper & Vector Ingestion Engine
├── web_app.py          # Online Streamlit Multi-Page Chat Application
└── README.md           # Documentation & Pitch Deck Guide
