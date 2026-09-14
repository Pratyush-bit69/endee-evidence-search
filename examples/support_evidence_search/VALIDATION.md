# Validation status

New project created September 14, 2026 with AI coding assistance.

- Local pure-Python tests: all nine passed on September 14, 2026.
- Real Endee server, embedding model and integration evaluation: passed in
  [GitHub Actions run 34820924000](https://github.com/Pratyush-bit69/endee-evidence-search/actions/runs/34820924000)
  for commit `a6be870` on September 14, 2026. The runner built this fork's native
  database server, ingested real BGE embeddings and performed filtered retrieval.
- All ten synthetic queries retrieved their labeled document first: Hit@1,
  Recall@3 and MRR@3 were each 1.0. This tiny, authored demonstration is **not**
  evidence of production accuracy or generalization. Full per-query citations
  and timings are in [ci-evaluation.json](results/ci-evaluation.json).
- The current local user cannot access the Docker daemon. No local database
  integration success is claimed; CI builds the fork's source on an isolated runner.
