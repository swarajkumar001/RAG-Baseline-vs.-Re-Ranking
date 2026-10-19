# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Compares two retrieval pipelines over the same chunked corpus of the EU GDPR (Regulation 2016/679):

- **Pipeline 1:** bi-encoder (`all-MiniLM-L6-v2`) → Top-10 by cosine similarity → Top-3 to the LLM
- **Pipeline 2:** same Top-10 → cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranking → Top-3 to the LLM

Everything is in [`rag_baseline_vs_rerank.ipynb`](rag_baseline_vs_rerank.ipynb):
sentence-aware chunking, both pipelines, an aggregate evaluation on 56 questions (Hit@k, MRR),
9 failure-mode case studies with rank-shift tables and LLM answers, latency profiling and a scaling experiment.

## Files

| File | Content |
|---|---|
| `rag_baseline_vs_rerank.ipynb` | the full experiment, with outputs |
| `data/gdpr.pdf` | official GDPR text (Official Journal L 119, 4.5.2016, EU Publications Office) |
| `data/eval_queries.json` | 56 evaluation questions, each with an evidence phrase quoted from the GDPR |
| `data/llm_answers.json` | cached LLM answers, so the notebook re-runs without an API key |

## Run

Tested with Python 3.14 on Windows (CPU only).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

set GEMINI_API_KEY=your-key       # optional; Linux/macOS: export GEMINI_API_KEY=your-key
jupyter notebook rag_baseline_vs_rerank.ipynb
```

LLM: `gemma-4-31b-it` (Google's open-weights Gemma 4, called through the Gemini API, temperature 0).
Set `USE_LLM_CACHE = False` in the notebook to regenerate the answers instead of using the cache.
