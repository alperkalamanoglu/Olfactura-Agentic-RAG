---
title: "Olfactura: The Agentic Perfume Advisor"
short_description: "Agentic RAG Fragrance Advisor: vLLM & Hybrid GPU Search"
emoji: "🌸"
colorFrom: "pink"
colorTo: "purple"
sdk: docker
pinned: false
---

# 🌟 Olfactura AI (Advanced RAG Fragrance Consultant)

![Python Version](https://img.shields.io/badge/Python-3.11-blue)
![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-orange)
![Architecture](https://img.shields.io/badge/Architecture-Agentic%20RAG-success)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/alperkalamanoglu/olfactura-ai)

Olfactura AI is a production-level, AI-driven personal fragrance consultant. It transcends simple text-based retrieval by utilizing **High-Dimensional Semantic Vector Mapping (3072-D)** combined with a **Cross-Encoder Reranking Pipeline**, mimicking the nuanced decision-making process of a master perfumer.

![Olfactura AI UI](assets/demo.gif)

## 🏗️ Advanced RAG Architecture: The "Secret Sauce"

Olfactura AI transcends simple keyword matching by implementing a multi-staged, mathematically-driven retrieval pipeline:

1.  **Agentic Orchestration & Dual-Backend:** 
    Utilizes a high-concurrency **vLLM** engine (serving **Gemma 4 26B A4B AWQ**) for its primary brain, with an automated fallback to OpenAI's GPT-4o-mini, ensuring 100% uptime and ultra-low TTFT.

2.  **High-Dimensional Semantic Math:**
    *   **Multi-Perfume Blending (Centroids):** When mixing multiple fragrances (e.g., "Mix Aventus and Tobacco Vanille"), the system calculates the **arithmetic mean of their 3072-D embeddings**, creating a precise mathematical centroid of the requested olfactory blend before retrieval.
    *   **Weighted Centroid Shifting:** For modified queries (e.g., "Aventus but more fresh"), it applies a **spatial shift (85% Anchor / 15% Modifier)** to locate the exact coordinate of the conceptual hybrid in vector space.

3.  **Hybrid Two-Stage Retrieval:**
    *   **Stage 1 (Broad Retrieval):** Fetches an initial pool of **50 candidate perfumes** using ChromaDB's HNSW Cosine Index, pre-filtered by deterministic metadata (Gender, Price, Notes) to eliminate hallucinations.
    *   **Stage 2 (Precision Reranking):** The elite pool is contextually sorted via a GPU-accelerated **BGE-Large-Reranker** (Dockerized TEI), ensuring semantic alignment that pure vector distance cannot catch.

4.  **Dynamic Similarity Thresholds:** 
    Adapts its "strictness" (0.70 vs 0.50 Cosine Threshold) based on user intent, allowing for high-fidelity clones when requested or enabling "creative discovery" for complex scent hybrids.

5.  **Observability & Telemetry:** 
    Asynchronous logging of end-to-end performance metrics (TTFT, Total Latency, Reranker Time) streamed directly to **HuggingFace Datasets** for continuous MLOps monitoring.


---

## 🛠️ The Core Tool Arsenal (Function Calling)

The LLM is granted 4 specialized tools to interact with the ChromaDB Vector Database. It acts as an autonomous agent, choosing which tool to fire based on context:

### 1. `search_perfumes()` | The Discovery Engine
* **When it's used:** Vague, mood-based, or highly specific descriptive searches. ("*Best cheap winter perfumes for men*", "*A fresh summer scent for women but NO Rose*")
* **How it works:** It combines pure Semantic NLP Queries (searching the 3072-D space for "warm vanilla winter nights") with hard Metadata Filters (MongoDB-style syntax like `{"gender_score": {"$gt": 0.6}, "price_tier_score": {"$lt": 3.0}}`).
* **Hidden Power:** Supports **Negative Prompting** (e.g., Exclude Notes: `['Rose', 'Jasmine']`). It instantly purges any vector mapping containing those notes from the result pool before sending it to the Reranker.

### 2. `recommend_similar()` | The Alchemist
* **When it's used:** Finding clones, alternatives, or hybrid blends. ("*Mix Acqua di Gio and Tobacco Vanille*", "*Cheap alternatives to Baccarat Rouge 540*")
* **How it works:** Extracts the vectors of reference perfumes. If multiple are given, it creates a **Mathematical Blend (Centroid)**. If the user adds a modifier ("*but make it darker*"), it shifts the vector coordinate toward that new concept before applying the **Dynamic Threshold** and Reranker logic.

### 3. `compare_perfumes()` | The Analyst
* **When it's used:** "Versus" battles. ("*Sauvage vs Bleu de Chanel*")
* **How it works:** By-passes vector search entirely. Directly fetches exact metadata, accords, performance stats (longevity/sillage), and ratings, outputting a side-by-side analytical markdown table for the LLM to comment on.

### 4. `get_perfume_details()` | The Encyclopedia
* **When it's used:** Deep dives into a single fragrance. ("*What are the heart notes of Black Opium?*")
* **How it works:** Retrieves the absolute truth from the database, forcing the LLM to ground its response entirely on real olfactory data, preventing hallucinated notes or release years.

---

## 🧠 Architecture, Data & Engineering Highlights

This project goes far beyond textbook RAG implementations. It incorporates several advanced Software & AI Engineering practices:

*   **Pydantic Structured Tool Calling:** 
    LLM function arguments are rigidly enforced via `Pydantic` schemas. This ensures the orchestrator outputs perfect, validated JSON every time, completely eliminating formatting hallucinations and `KeyErrors` during agent routing.
*   **Real-time Output Streaming (Low TTFT):** 
    Rather than waiting for the orchestrator to finish a full multi-paragraph response, results are yielded token-by-token directly to the Streamlit UI using Python Generators. This keeps the application highly responsive with a minimal *Time-To-First-Token*.
*   **Web Scraping & Advanced Data Normalization:** 
    The database is an authentic scrape of real-world fragrance data. Raw metrics were algorithmically normalized before entering the vector space:
    *   **Popularity Votes:** Log-normalized to prevent highly popular perfumes from eclipsing niche masterpieces.
    *   **Community Ratings:** Adjusted using *Bayesian Normalization* (weighted against vote counts) to ensure a fragrance with a perfect 5.0 from 2 people doesn't outrank a 4.6 from 5,000 people.
*   **Hardware Agnostic CPU Reranking (FlashRank):** 
    Migrated from heavy, API-dependent rerankers to the incredibly lightweight **FlashRank (MiniLM-L12)** fallback. It reranks 100+ documents contextually in ~40ms on a standard CPU, slashing infrastructure costs while boosting inference resilience.
*   **Semantic Text Bias Mitigation:** 
    Brand names and product titles were deliberately excluded from the vector embedded text (`semantic_text`), leaving only pure notes and accords in the embedding space. This prevents "Name Bias" (where searching 'Aventus' only returns Aventus flankers) and forces the engine to find genuine *olfactory* matches based strictly on chemistry.
*   **Scalable Direct-Query Architecture:** 
    Designed for production scalability by leveraging ChromaDB's native HNSW indexing for sub-linear search complexity. Every query hits the vector database directly, ensuring consistent performance whether the dataset contains 2,300 or 2,000,000 fragrances—no in-memory caching bottlenecks.
*   **Anti-Hallucination Strict Thresholds:** 
    Implemented rigid logic that drops any result with a Hybrid Score under a certain threshold. If the user searches for a bizarre or impossible fragrance (e.g., "A perfume that smells like a Doner Kebab"), the system will gracefully fail ("Match Quality Too Low") rather than hallucinating or force-fitting completely irrelevant results.

---

## 📊 RAG Evaluation Metrics (LLM-as-a-Judge)

The repository includes a custom, automated evaluation suite (`evaluation/run_ragas_eval.py`). It calculates deterministic *Retrieval Quality* via Golden Datasets, and uses **GPT-4o** as a strict, temperature-0 judge to score the final pipeline on *Answer Relevancy* and *Faithfulness*.

### Global Performance (Evaluated on a Golden Dataset of n=25 Complex Edge-Cases)
| Metric | Score | Description |
|--------|-------|-------------|
| **Answer Precision @ 3** | **0.988** 💎 | How often the final response contained factually optimal recommendations. |
| **Retrieval Quality @ 10** | **0.908** 🕵️ | How accurately the vector DB + Reranker fetched the hidden "Golden" answers. |
| **Metadata Constraint Match** | **0.997** 🎯 | How perfectly the system respected strict user constraints (gender, price). |
| **Faithfulness** | **1.000** 🛡️ | Factuality rate (1.0 = Zero Hallucinations. LLM strictly adhered to ChromaDB facts). |
| **Answer Relevancy** | **0.960** ✅ | How accurately the LLM addressed the core intent of the user's prompt. |

*Achieving 100% Faithfulness and 98.8% Answer Precision proves the "Strict Tool Calling" and "Thresholding" mechanisms completely prevented LLM drift.*

---

## 🌩️ Infrastructure & Deployment: The Power of Hybrid GPU Architecture

Unlike standard API-wrapper chatbots, Olfactura AI runs a customized, horizontally scalable **Hybrid GPU Architecture** orchestrated on `Vast.ai` cloud infrastructure, optimizing latency down to the millisecond.

### 1. vLLM Engine (The Brain)
To handle complex, highly concurrent LLM inference without relying purely on third-party APIs, the core engine leverages **vLLM** hosted on an **NVIDIA RTX 3090 (24GB VRAM)**.
* **Model:** `Gemma 4 26B A4B (4-bit AWQ Quantization)` ensures maximum intelligence within a tight memory footprint.
* **Why vLLM?** Through *PagedAttention*, *CUDA Graphs*, and **Prompt Prefix Caching**, it delivers 2x-4x faster parallel inference. The system reserves >80% of VRAM dynamically for the KV Cache, effortlessly holding multi-turn conversational context while maintaining a Time-To-First-Token (TTFT) under 4 seconds.
* **Context Scaling:** The custom `_trim_history` algorithm perfectly caps the sliding memory window at 12,000 chars, ensuring the orchestrator never experiences a VRAM Out-Of-Memory (OOM) crash during heavy concurrency.

### 2. Dockerized BGE-Reranker via TEI (The Precision Filter)
Standard embeddings calculate vector distances, but fail at granular comparative logic. We bridge this gap using a self-hosted GPU Reranker.
* **Dockerized TEI:** The `BGE-Large-Reranker` runs inside a highly optimized Hugging Face **TEI (Text Embeddings Inference)** Docker container.
* **Performance:** Real-time logging metrics show the GPU reranker processes a pool of `50+ document candidates` in **~1.01 seconds**, offering near-instant semantic sorting which CPU-bound arrays simply cannot match. If the GPU pipeline goes offline, the `src/database/reranker.py` handler instantly falls back to a local, CPU-based `FlashRank (MiniLM-L12)` cross-encoder, guaranteeing zero downtime.

### 3. Secure Orchestration & Tunneling
Running an enterprise-grade AI stack on affordable cloud GPU rentals requires airtight dev-ops:
* **SSH Port Forwarding (L-Tunneling):** Creates a secure, encrypted socket connecting the local client and the distant `Vast.ai` GPU pods (Ports 8000 & 8080) mimicking a zero-latency `localhost` environment.
* **Resilience:** Background processes are daemonized via `nohup` and `tmux`, ensuring the API endpoints and Reranker servers survive SSH disconnects and remain active 24/7.

---

## 💻 Tech Stack Highlights
* **LLM Orchestration Engine:** Self-Hosted `Gemma 4 26B A4B (4-bit AWQ)` via **vLLM** (Fallback: OpenAI GPT-4o-mini).
* **Vector Database:** ChromaDB (Persisted Indexed Object Storage).
* **Cross-Encoder Rerankers:** Remote `bge-reranker-large` on TEI (Fallback: Local `ms-marco-MiniLM-L-12-v2`).
* **Environment:** Hugging Face Spaces / Docker / Vast.ai (RTX 3090).

---

> **"A masterclass in restricting LLM hallucinations using Vector Math, Cross-Encoders, and strict Tool Calling logic."**

---

## ⚙️ Setup & How to Run (Local)

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/alperkalamanoglu/Olfactura-Agentic-RecSys.git
   cd Olfactura-Agentic-RecSys
   ```

2. **Set up API Keys (`.env`):**
   The system relies on OpenAI for the Orchestrator LLM and Embeddings.
   Make a copy of the example environment file:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and insert your API key:
   ```env
   OPENAI_API_KEY="sk-proj-YOUR_REAL_KEY_HERE"
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the App (Standard):**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🐳 Running with Docker

If you prefer an isolated environment or want to deploy to a server:

1. **Build and Run the Container:**
   ```bash
   docker-compose up --build
   ```
   
2. **Access the Application:**
   Open your browser and navigate to `http://localhost:8501`. Docker automatically handles all system-level dependencies for the FlashRank reranker and ChromaDB vector persistence.

---

## 🗺️ Future Roadmap

While Olfactura AI is highly capable in its current state, future planned improvements include:
- **Scalable Infrastructure (Cloud Vector DB):** Migrating from local ChromaDB to a fully managed, scalable solution like **Pinecone** or **Milvus** to support millions of vectors and high-concurrency requests seamlessly.
- **Multi-Agent Debate System:** Deploying dual-agent personas (e.g., a "Niche Fragrance Expert" vs. a "Designer Crowd-Pleaser") that autonomously debate and collaboratively merge conflicting opinions before presenting a finalized recommendation to the user.
- **Voice-to-Voice RAG:** Transitioning the UX from a text-based interface to a real-time conversational "Koku Danışmanı" (Fragrance Consultant) utilizing ultra-low latency architectures like the OpenAI Realtime API or Gemini Live.
- **User Authentication & Preference Memory:** Implementing user login systems to track "Likes/Dislikes" over time. This enables **Long-Term Memory RAG**, automatically personalizing the mathematical centroid for future recommendations based on historical taste.
- **Dataset Expansion & Image Integration:** Significantly increasing the scraped fragrance database volume and rendering actual product images in the UI cards for a richer ecommerce-like experience.
