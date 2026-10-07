"""Weaviate access: collections, ingestion, hybrid search, neighbour lookup.

Adapted from interface/chunks/collection.py, interface/chunks/ingest.py and
interface/retrieval/search.py, simplified to one text field per chunk and
Weaviate's native hybrid (BM25 + vector) search.

Storage: one multi-tenant collection holds full emails (kind="email", no vector) and their
searchable chunks (kind="chunk", with a vector); every query filters on kind.

Isolation model:
  * tenant  = whose data it is. The collection uses Weaviate multi-tenancy; each user has
              their own tenant, and the Enron emails live in the shared "public" tenant.
              Every read and write goes through exactly one tenant.
  * dataset = a named set of emails inside a tenant (e.g. "enron", "my-mailbox"),
              stored as a property and used as a filter.
"""

import atexit
import threading

import weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from forensic_rag import config
from forensic_rag.models import get_embeddings

CHUNK_PROPS = ["email_id", "dataset", "chunk_id", "chunk_idx", "text", "subject"]
EMAIL_PROPS = ["email_id", "dataset", "subject", "sender", "recipients", "date", "docID", "body"]

_client: weaviate.WeaviateClient | None = None
_client_lock = threading.Lock()


def get_client() -> weaviate.WeaviateClient:
    """One shared client; the lock matters because the MCP server calls this from worker threads."""
    global _client
    with _client_lock:
        if _client is None:
            if config.WEAVIATE_URL:
                _client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=config.WEAVIATE_URL,
                    auth_credentials=Auth.api_key(config.WEAVIATE_API_KEY),
                )
            else:
                _client = weaviate.connect_to_local(
                    host=config.WEAVIATE_HOST,
                    port=config.WEAVIATE_PORT,
                    grpc_port=config.WEAVIATE_GRPC_PORT,
                )
            atexit.register(_client.close)
        return _client


# --------------------------------------------------------------------------- #
# Collection and tenants
# --------------------------------------------------------------------------- #

def _field(name: str) -> Property:
    # FIELD tokenization -> exact-match filtering
    return Property(name=name, data_type=DataType.TEXT, tokenization=Tokenization.FIELD)


def create_collections(reset: bool = False) -> None:
    client = get_client()

    if reset:
        for name in (config.COLLECTION, *config.LEGACY_COLLECTIONS):
            if client.collections.exists(name):
                client.collections.delete(name)

    if not client.collections.exists(config.COLLECTION):
        client.collections.create(
            name=config.COLLECTION,
            vector_config=Configure.Vectors.self_provided(name=config.VECTOR_NAME),
            multi_tenancy_config=Configure.multi_tenancy(
                enabled=True, auto_tenant_creation=True, auto_tenant_activation=True),
            properties=[
                _field("kind"),                                   # "email" or "chunk"
                _field("dataset"),
                Property(name="email_id", data_type=DataType.INT),
                Property(name="subject", data_type=DataType.TEXT),
                # chunk objects
                _field("chunk_id"),
                Property(name="chunk_idx", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
                # email objects
                Property(name="sender", data_type=DataType.TEXT),
                Property(name="recipients", data_type=DataType.TEXT),
                Property(name="date", data_type=DataType.TEXT),
                Property(name="docID", data_type=DataType.TEXT),
                Property(name="body", data_type=DataType.TEXT),
            ],
        )


def _coll(tenant: str):
    return get_client().collections.get(config.COLLECTION).with_tenant(tenant)


def _tenant_exists(tenant: str) -> bool:
    client = get_client()
    return (client.collections.exists(config.COLLECTION)
            and client.collections.get(config.COLLECTION).tenants.exists(tenant))


def _kind(kind: str):
    return Filter.by_property("kind").equal(kind)


def _ds_filter(dataset: str):
    return Filter.by_property("dataset").equal(dataset)


def delete_dataset(tenant: str, dataset: str) -> None:
    """Remove all objects (emails and chunks) of one dataset in one tenant."""
    if _tenant_exists(tenant):
        _coll(tenant).data.delete_many(where=_ds_filter(dataset))


def list_datasets(tenant: str) -> list[dict]:
    """[{dataset, emails}] for every dataset in one tenant."""
    if not _tenant_exists(tenant):
        return []
    res = _coll(tenant).aggregate.over_all(
        group_by=GroupByAggregate(prop="dataset"), total_count=True, filters=_kind("email")
    )
    return sorted(({"dataset": g.grouped_by.value, "emails": g.total_count} for g in res.groups),
                  key=lambda d: d["dataset"])


def dataset_exists(tenant: str, dataset: str) -> bool:
    return _tenant_exists(tenant) and count("email", tenant, dataset) > 0


def count(kind: str, tenant: str, dataset: str | None = None) -> int:
    """Number of "email" or "chunk" objects in a tenant (optionally one dataset)."""
    filters = _kind(kind) & _ds_filter(dataset) if dataset else _kind(kind)
    return _coll(tenant).aggregate.over_all(total_count=True, filters=filters).total_count


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #

def insert_emails(tenant: str, emails: list[dict]) -> None:
    collection = _coll(tenant)
    with collection.batch.dynamic() as batch:
        for e in emails:
            batch.add_object(
                properties={**e, "kind": "email"},
                uuid=generate_uuid5({"kind": "email", "dataset": e["dataset"], "email_id": e["email_id"]}),
            )
    _raise_on_failed(collection)


def insert_chunks(tenant: str, chunks: list[dict], vectors: list[list[float]]) -> None:
    collection = _coll(tenant)
    with collection.batch.dynamic() as batch:
        for c, v in zip(chunks, vectors, strict=True):
            batch.add_object(
                properties={**c, "kind": "chunk"},
                vector={config.VECTOR_NAME: v},
                uuid=generate_uuid5({"kind": "chunk", "dataset": c["dataset"], "chunk_id": c["chunk_id"]}),
            )
    _raise_on_failed(collection)


def _raise_on_failed(collection) -> None:
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} objects failed to insert, first: {failed[0].message}")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def hybrid_search(tenant: str, query: str, dataset: str, k: int = 10, alpha: float = 0.5) -> list[dict]:
    """Weaviate hybrid search over chunk text.

    alpha = 0 -> pure BM25, alpha = 1 -> pure vector search.
    """
    vector = get_embeddings().embed_query(query)
    response = _coll(tenant).query.hybrid(
        query=query,
        vector=vector,
        target_vector=config.VECTOR_NAME,
        alpha=alpha,
        limit=k,
        query_properties=["text"],
        filters=_kind("chunk") & _ds_filter(dataset),
        return_properties=CHUNK_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    return [{**o.properties, "score": float(o.metadata.score or 0.0)} for o in response.objects]


def get_neighbors(tenant: str, dataset: str, email_id: int, chunk_idx: int, window: int = 1) -> list[dict]:
    """Chunks chunk_idx-window ... chunk_idx+window of one email, in order."""
    filters = (
        _kind("chunk")
        & _ds_filter(dataset)
        & Filter.by_property("email_id").equal(int(email_id))
        & Filter.by_property("chunk_idx").greater_or_equal(chunk_idx - window)
        & Filter.by_property("chunk_idx").less_or_equal(chunk_idx + window)
    )
    response = _coll(tenant).query.fetch_objects(filters=filters, limit=2 * window + 1,
                                                 return_properties=CHUNK_PROPS)
    return sorted((o.properties for o in response.objects), key=lambda p: p["chunk_idx"])


def get_email(tenant: str, dataset: str, email_id: int) -> dict | None:
    filters = _kind("email") & _ds_filter(dataset) & Filter.by_property("email_id").equal(int(email_id))
    response = _coll(tenant).query.fetch_objects(filters=filters, limit=1, return_properties=EMAIL_PROPS)
    return response.objects[0].properties if response.objects else None
