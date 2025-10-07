from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI, APIError
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
import os, re, logging, random, tiktoken
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from dotenv import load_dotenv
from traceback import format_exc
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CORS_ORIGINS = [o.strip() for o in (os.getenv("CORS_ORIGINS","https://addictiontube.com").split(","))]

missing = [k for k,v in dict(OPENAI_API_KEY=OPENAI_API_KEY, WEAVIATE_CLUSTER_URL=WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY=WEAVIATE_API_KEY).items() if not v]
if missing:
    raise EnvironmentError("Missing env vars: " + ", ".join(missing))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

logger = logging.getLogger("faq-backend")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('/tmp/faq_search.log', maxBytes=10_485_760, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(console)

limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "60 per hour"], storage_uri="memory://", headers_enabled=True)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_weaviate_client():
    for attempt in range(3):
        try:
            wc = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_CLUSTER_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
                headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
                skip_init_checks=False
            )
            if wc.is_ready():
                return wc
            wc.close()
        except Exception as e:
            logger.warning(f"Weaviate init attempt {attempt+1} failed: {e}")
    raise EnvironmentError("Weaviate client initialization failed")

def get_embedding(text):
    resp = client.embeddings.create(input=text, model=EMBED_MODEL)
    return resp.data[0].embedding

def strip_query(q: str) -> str:
    return re.sub(r'[\r\n\t]+',' ', q).strip()

@app.route("/", methods=["GET","HEAD"])
def healthz():
    wc = get_weaviate_client()
    try:
        ok = wc.is_ready()
        colls = wc.collections.list_all()
        return {"ok": ok, "has_FAQ": "FAQ" in colls}, 200 if ok else 503
    finally:
        wc.close()

@app.errorhandler(429)
def oops(e):
    return jsonify({"error":"rate_limited", "details":str(e.description)}), 429

@app.route("/search_faq", methods=["POST"])
@limiter.limit("120/hour")
def search_faq():
    data = request.get_json(force=True) or {}
    query = strip_query(data.get("query",""))
    if not query:
        return jsonify({"error":"query required"}), 400
    top_k = int(data.get("top_k", 8))
    category = (data.get("category") or "").strip()
    subcategory = (data.get("subcategory") or "").strip()
    tags = data.get("tags") or []

    wc = get_weaviate_client()
    try:
        vector = get_embedding(query)
        col = wc.collections.get("FAQ")

        f = None
        def andf(a,b): return a & b if a and b else (a or b)
        if category:
            f = andf(f, Filter.by_property("category").equal(category))
        if subcategory:
            f = andf(f, Filter.by_property("subcategory").equal(subcategory))
        if tags:
            f = andf(f, Filter.by_property("tags").contains_any(tags))

        res = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=f,
            return_metadata=["distance"],
            return_properties=["faq_id","question","answer","category","subcategory","tags","source","created_at","updated_at"]
        )

        out = []
        for o in res.objects:
            p, m = o.properties or {}, o.metadata or {}
            out.append({
                "distance": m.get("distance"),
                "score": 1 - m.get("distance") if m.get("distance") else None,
                "faq_id": p.get("faq_id"),
                "question": p.get("question"),
                "answer": p.get("answer"),
                "category": p.get("category"),
                "subcategory": p.get("subcategory"),
                "tags": p.get("tags"),
                "source": p.get("source"),
                "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at")
            })
        return jsonify({"results": out})
    finally:
        wc.close()

@app.route("/rag_faq", methods=["GET"])
@limiter.limit("60/hour")
def rag_faq():
    q = strip_query(request.args.get("q",""))
    if not q:
        return jsonify({"error":"missing q"}), 400
    top_k = int(request.args.get("k","8"))
    reroll = request.args.get("reroll","").lower().startswith("y")

    wc = get_weaviate_client()
    try:
        vector = get_embedding(q)
        col = wc.collections.get("FAQ")
        res = col.query.near_vector(
            near_vector=vector,
            limit=max(20, top_k),
            return_metadata=["distance"],
            return_properties=["faq_id","question","answer","category","subcategory","tags"]
        )
        matches = res.objects or []
        if not matches:
            return jsonify({"error":"no matches found"}), 404
        if reroll:
            random.shuffle(matches)

        # Build context with a robust char budget (no tiktoken needed)
        CHAR_BUDGET = 24000  # roughly ~12k tokens equivalent margin
        used = 0
        ctx_parts = []
        sources = []

        for o in matches:
            p = (o.properties or {})
            m = (o.metadata or {})
            fid = str(p.get("faq_id") or "").strip()
            qtxt = (p.get("question") or "").strip()
            atxt = (p.get("answer") or "").strip()
            piece = f"FAQ:{fid}\nQ: {qtxt}\nA: {atxt}\n"
            if used + len(piece) > CHAR_BUDGET:
                break
            ctx_parts.append(piece)
            used += len(piece)
            if fid:
                dist = m.get("distance")
                sources.append({
                    "faq_id": fid,
                    "distance": dist,
                    "score": (1 - dist) if isinstance(dist, (int, float)) else None
                })
            if len(sources) >= top_k:
                break

        context = "\n---\n".join(ctx_parts)
        allowed_ids = ", ".join(s["faq_id"] for s in sources if s.get("faq_id"))

        system = (
            "You are an empathetic addiction-recovery FAQ assistant. "
            "Answer clearly, briefly, and supportively using ONLY the provided FAQs. "
            f"When citing, ONLY use IDs from this set: [{allowed_ids}]. "
            "Citations must appear inline like [FAQ:123]; you may include multiple."
        )
        user = (
            "Use ONLY the FAQs below as ground truth. Then answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {q}\n"
            "Answer succinctly (120-200 words), with inline citations [FAQ:<id>] that are actually in the provided list."
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                max_tokens=700,
                temperature=0.3
            )
            answer = (resp.choices[0].message.content or "").strip()

            # If model forgot citations, append a Sources line so users still see provenance
            if "[FAQ:" not in answer and sources:
                src_line = "Sources: " + ", ".join(f"[FAQ:{s['faq_id']}]" for s in sources[:top_k])
                answer = f"{answer}\n\n{src_line}"

            return jsonify({"answer": answer, "sources": sources[:top_k]})
        except Exception as e:
            logger.exception(f"OpenAI call failed: {e}")
            return jsonify({"error":"openai_unavailable","details":str(e)}), 502

    except Exception as e:
        logger.exception(f"/rag_faq failed: {e}\n{format_exc()}")
        return jsonify({"error":"internal_error","details":str(e)}), 500
    finally:
        wc.close()



# -----------------------------------------
# NEW: FAQ STATS ROUTE
# -----------------------------------------
@app.route("/stats_faq", methods=["GET"])
@limiter.limit("30/minute")
def stats_faq():
    wc = get_weaviate_client()
    try:
        col = wc.collections.get("FAQ")
        result = col.aggregate.over_all(total_count=True, group_by="category")
        total = result.total_count
        groups = []
        for g in result.groups or []:
            groups.append({"category": g.value or "(Uncategorized)", "count": g.total_count})
        q_latest = col.query.fetch_objects(limit=1, return_properties=["updated_at"], sort=[{"path": ["updated_at"], "order": "desc"}])
        latest = None
        if q_latest.objects:
            latest = q_latest.objects[0].properties.get("updated_at")

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_count": total,
            "categories": groups,
            "last_updated_at": latest
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error":"stats_failed","details":str(e)}), 500
    finally:
        wc.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT","8002"))
    logger.info(f"AddictionTube FAQ backend started on port {port}")
    app.run(host="0.0.0.0", port=port)
