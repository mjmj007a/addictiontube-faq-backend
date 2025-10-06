import os, pymysql, logging, json
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

# Env
MYSQL_HOST=os.getenv("MYSQL_HOST","localhost")
MYSQL_DB=os.getenv("MYSQL_DB")
MYSQL_USER=os.getenv("MYSQL_USER")
MYSQL_PASS=os.getenv("MYSQL_PASS")

WEAVIATE_CLUSTER_URL=os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY=os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
EMBED_MODEL=os.getenv("EMBED_MODEL","text-embedding-3-small")
BATCH_SIZE=int(os.getenv("BATCH_SIZE","128"))
CLASS_NAME="FAQ"
VECTOR_DIM=1536

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("faq-indexer")

# Clients
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_weaviate = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    skip_init_checks=False
)

def mysql_conn():
    return pymysql.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS,
        database=MYSQL_DB, charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor
    )

def ensure_schema():
    collections = client_weaviate.collections.list_all()
    if CLASS_NAME not in collections:
        client_weaviate.collections.create(
            name=CLASS_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="faq_id", data_type=DataType.TEXT, index_filterable=True, index_searchable=True),
                Property(name="question", data_type=DataType.TEXT, index_searchable=True),
                Property(name="answer", data_type=DataType.TEXT, index_searchable=True),
                Property(name="category", data_type=DataType.TEXT, index_filterable=True, index_searchable=True),
                Property(name="subcategory", data_type=DataType.TEXT, index_filterable=True, index_searchable=True),
                Property(name="tags", data_type=DataType.TEXT_ARRAY, index_filterable=True, index_searchable=True),
                Property(name="source", data_type=DataType.TEXT, index_filterable=True),
                Property(name="created_at", data_type=DataType.DATE, index_filterable=True),
                Property(name="updated_at", data_type=DataType.DATE, index_filterable=True),
            ]
        )
        log.info("Created Weaviate collection 'FAQ' (vectorizer: none)")
    else:
        log.info("Weaviate collection 'FAQ' exists")

def norm_tags(raw):
    if not raw: return []
    if isinstance(raw, list): return [t.strip() for t in raw if t.strip()]
    return [t.strip() for t in str(raw).replace(";",",").split(",") if t.strip()]

def build_text(row:Dict)->str:
    return f"Q: {row['question'].strip()}\nA: {row['answer'].strip()}\nCategory: {row.get('category') or ''}\nSubcategory: {row.get('subcategory') or ''}\nTags: {', '.join(norm_tags(row.get('tags')))}"

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def embed_batch(texts:List[str]):
    resp = client_openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def fetch_rows(since_iso:str=None)->List[Dict]:
    q = "SELECT id AS faq_id, question, answer, category, subcategory, tags, source, created_at, updated_at FROM addiction_faqs"
    params=[]
    if since_iso:
        q += " WHERE updated_at >= %s"
        params=[since_iso]
    q += " ORDER BY id ASC"
    with mysql_conn() as conn, conn.cursor() as cur:
        cur.execute(q, params)
        return cur.fetchall()

def upsert_batch(rows:List[Dict]):
    col = client_weaviate.collections.get(CLASS_NAME)
    texts = [build_text(r) for r in rows]
    vecs = embed_batch(texts)
    with col.batch.dynamic() as batch:
        for r, v in zip(rows, vecs):
            props = {
                "faq_id": str(r["faq_id"]),
                "question": r["question"] or "",
                "answer": r["answer"] or "",
                "category": (r.get("category") or "").strip(),
                "subcategory": (r.get("subcategory") or "").strip(),
                "tags": norm_tags(r.get("tags")),
                "source": (r.get("source") or "AddictionTube FAQ").strip(),
                "created_at": (r.get("created_at") or datetime.utcnow()).isoformat(),
                "updated_at": (r.get("updated_at") or datetime.utcnow()).isoformat(),
            }
            batch.add_object(properties=props, uuid=generate_uuid5(r["faq_id"]), vector=v)

if __name__ == "__main__":
    ensure_schema()
    since = os.getenv("LAST_SYNC_UTC") or None
    rows = fetch_rows(since)
    if not rows:
        log.info("No FAQs to index")
    else:
        for i in range(0, len(rows), int(os.getenv("BATCH_SIZE","128"))):
            chunk = rows[i:i+int(os.getenv("BATCH_SIZE","128"))]
            upsert_batch(chunk)
            log.info(f"Upserted {i+len(chunk)}/{len(rows)}")
    client_weaviate.close()
