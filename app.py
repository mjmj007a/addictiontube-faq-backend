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
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CORS_ORIGINS = [o.strip() for o in (os.getenv("CORS_ORIGINS","https://addictiontube.com").split(","))]

# Basic validation
missing = [k for k,v in dict(OPENAI_API_KEY=OPENAI_API_KEY, WEAVIATE_CLUSTER_URL=WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY=WEAVIATE_API_KEY).items() if not v]
if missing:
    raise EnvironmentError("Missing env vars: " + ", ".join(missing))

# App + CORS + Limiter + Logging
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

logger = logging.getLogger("faq-backend")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('/tmp/faq_search.log', maxBytes=10_485_760, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)

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
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
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
        if reroll:
            random.shuffle(matches)

        if not matches:
            return jsonify({"error":"no matches found"}), 404

        # Build context window
        enc = tiktoken.get_encoding("cl100k_base")
        max_tokens = 12000
        ctx, used = [], 0
        for o in matches:
            p = o.properties or {}
            chunk = f"Q: {p.get('question','')}
A: {p.get('answer','')}"
            t = len(enc.encode(chunk))
            if used + t > max_tokens: break
            ctx.append(chunk); used += t
        context = "\n\n---\n\n".join(ctx)

        system = "You are an empathetic addiction-recovery FAQ assistant. Answer clearly and cite specific FAQ ids you used."
        user = f"Use the following FAQs as your only ground truth. Then answer the question.

{context}

Question: {q}
Answer:"
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                max_tokens=700
            )
            answer = resp.choices[0].message.content
            return jsonify({"answer": answer})
        except APIError as e:
            return jsonify({"error":"openai_unavailable","details":str(e)}), 502
    finally:
        wc.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT","8002"))
    app.run(host="0.0.0.0", port=port)
