# AddictionTube FAQ – Phase 3 (Semantic Search)

This repo contains:
- `setup_and_import_faqs.py` – MySQL → OpenAI embeddings → Weaviate (manual vectors) for FAQs
- `app.py` – Flask API on Render: `/search_faq` and `/rag_faq`
- `requirements.txt`, `.env.example`, `render.yaml`

## Deploy (Render)
1. Push to GitHub
2. Create a new Web Service in Render → connect repo
3. Render will use `render.yaml`
4. Add env vars from `.env.example` (do NOT commit real secrets)

## First full index
```
pip install -r requirements.txt
cp .env.example .env  # fill secrets
python setup_and_import_faqs.py
```

Delta reindex:
```
LAST_SYNC_UTC="2025-10-06T00:00:00Z" python setup_and_import_faqs.py
```

## API
### POST /search_faq
```
{
  "query": "how do I prevent relapse after detox?",
  "top_k": 8,
  "category": "Relapse Prevention",
  "subcategory": "Triggers",
  "tags": ["CBT","support"]
}
```

### GET /rag_faq?q=...&reroll=yes
Returns a grounded, empathetic answer using top FAQ matches as context.
