# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Compares two retrieval pipelines over the same chunked corpus of the EU GDPR (Regulation 2016/679):

- **Pipeline 1:** bi-encoder (`all-MiniLM-L6-v2`) → Top-10 by cosine similarity → Top-3 to the LLM
- **Pipeline 2:** same Top-10 → cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranking → Top-3 to the LLM

Everything is in [`rag_baseline_vs_rerank.ipynb`](rag_baseline_vs_rerank.ipynb): chunking, both pipelines,
failure-mode test cases with rank-shift tables, LLM answer comparison, latency profiling and a scaling experiment.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

set GEMINI_API_KEY=your-key       # Linux/macOS: export GEMINI_API_KEY=your-key
jupyter notebook rag_baseline_vs_rerank.ipynb
```

LLM: `gemini-3.6-flash` (temperature 0). Generated answers are cached in `data/llm_answers.json`, so the notebook
re-runs completely without an API key.

## Data

`data/gdpr.pdf` – official text from the EU Publications Office (Official Journal L 119, 4.5.2016).
