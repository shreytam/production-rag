"""redis-vl-backed SemanticCache. The ONLY module importing redis-vl, and only
lazily inside method bodies — constructing an instance requires neither Redis nor
the redis-vl package, so build_cache stays importable in the offline suite.

Schema per tier (one RediSearch index): a COSINE vector field plus tenant_id,
collection_id and doc_ids TAG fields and a payload text field. The doc_ids TAG is
the reverse index that makes per-document eviction a filtered delete. Per-key TTL
backstops the new-document blind spot.
"""

from __future__ import annotations

import json
from typing import Sequence

from cache.semantic_cache import norm_collection


class RedisVLSemanticCache:
    def __init__(self, *, index_name: str, settings) -> None:
        self.index_name = index_name
        self.settings = settings
        self._index = None  # lazily built SearchIndex

    def _get_index(self):
        if self._index is not None:
            return self._index
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        schema = IndexSchema.from_dict({
            "index": {"name": self.index_name, "prefix": f"{self.index_name}:",
                      "storage_type": "hash"},
            "fields": [
                {"name": "tenant_id", "type": "tag"},
                {"name": "collection_id", "type": "tag"},
                {"name": "doc_ids", "type": "tag", "attrs": {"separator": "|"}},
                {"name": "payload", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {
                    "dims": self.settings.embed_dimension, "distance_metric": "cosine",
                    "algorithm": "flat", "datatype": "float32"}},
            ],
        })
        index = SearchIndex(schema, redis_url=self.settings.redis_url)
        index.create(overwrite=False)
        self._index = index
        return index

    def _distance_threshold(self) -> float:
        # redis-vl ranges cosine DISTANCE in [0, 2]; distance = 1 - similarity.
        return 1.0 - float(self.settings.cache_similarity_threshold)

    def lookup(self, *, tenant_id, collection_id, embedding) -> dict | None:
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        index = self._get_index()
        flt = (Tag("tenant_id") == tenant_id) & \
              (Tag("collection_id") == norm_collection(collection_id))
        q = VectorQuery(vector=list(embedding), vector_field_name="vector",
                        return_fields=["payload", "vector_distance"], num_results=1,
                        filter_expression=flt)
        results = index.query(q)
        if not results:
            return None
        top = results[0]
        if float(top["vector_distance"]) > self._distance_threshold():
            return None
        return json.loads(top["payload"])

    def store(self, *, tenant_id, collection_id, embedding, payload, doc_ids) -> None:
        import numpy as np

        index = self._get_index()
        vec = np.array(list(embedding), dtype=np.float32).tobytes()
        data = {
            "tenant_id": tenant_id,
            "collection_id": norm_collection(collection_id),
            "doc_ids": "|".join(doc_ids) if doc_ids else "",
            "payload": json.dumps(payload),
            "vector": vec,
        }
        index.load([data], ttl=int(self.settings.cache_ttl_seconds))

    def invalidate_document(self, *, tenant_id, collection_id, doc_id) -> int:
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag

        index = self._get_index()
        flt = (Tag("tenant_id") == tenant_id) & \
              (Tag("collection_id") == norm_collection(collection_id)) & \
              (Tag("doc_ids") == doc_id)
        matches = index.query(FilterQuery(filter_expression=flt, return_fields=["id"]))
        keys = [m["id"] for m in matches]
        if keys:
            index.drop_keys(keys)
        return len(keys)
