# 🌸 Olfactura AI - Deployment & Operations Guide

This guide details the deployment architectures and operational instructions for the Olfactura AI agent. The application is built to be highly scalable, running efficiently on both Local GPUs (vLLM) and Cloud Environments (Hugging Face Spaces).

## Prerequisites
- Python 3.10+ OR Docker Desktop.
- OpenAI API Key (For fallback inference and embeddings).

## Option 1: Docker (Hugging Face Spaces Native)
This is the recommended approach for deploying to any cloud provider or Hugging Face. The Dockerfile correctly exposes port `7860`.
1. **Verify Database**: Ensure the `chroma_db` folder exists in the root directory.
2. **Configure**: Add your `OPENAI_API_KEY` to the cloud secrets or local `.env` file.
3. **Run**:
   ```bash
   docker-compose up --build
   ```

## Option 2: Local Execution (Development & GPU)
1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Application**:
   ```bash
   streamlit run streamlit_app.py
   ```
*Note: The agent dynamically detects if `VLLM_BASE_URL` is provided. If a local GPU is found, it uses high-concurrency vLLM logic with aggressive `_trim_history` optimizations. If not, it activates the GPT-4o-mini fallback seamlessly.*

## Quality Assurance & Evaluation
The project uses the **RAGAS** (Retrieval Augmented Generation Assessment) framework integrated with MLflow principles for continuous generation quality testing.
To run the automated test suite over the test dataset:
```bash
python evaluation/run_ragas_eval.py
```
This evaluates the agent on Answer Relevancy, Context Precision, and Faithfulness.

## Telemetry
User analytics, True Time-To-First-Token (TTFT), and Reranker metrics are continuously written to `agent_activity_logs/agent_activity.log`. Ensure this directory is writable by the running process in production.
