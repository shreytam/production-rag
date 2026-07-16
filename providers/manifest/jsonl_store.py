from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from core.types import DocManifest


class JsonlManifestStore:
    def __init__(self, manifest_dir: str = ".cache/manifest") -> None:
        self.manifest_dir = Path(manifest_dir)

    def _path_for(self, tenant_id: str, doc_id: str) -> Path:
        tenant_safe = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        doc_safe = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        return self.manifest_dir / tenant_safe / f"{doc_safe}.json"

    def load(self, tenant_id: str, doc_id: str) -> DocManifest | None:
        path = self._path_for(tenant_id, doc_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("tenant_id") != tenant_id or data.get("doc_id") != doc_id:
                return None
            return DocManifest.model_validate(data)
        except Exception:
            return None

    def save(self, manifest: DocManifest) -> None:
        path = self._path_for(manifest.tenant_id, manifest.doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(manifest.model_dump(), indent=2))
            os.replace(tmp, path)  # atomic
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    def delete(self, tenant_id: str, doc_id: str) -> None:
        path = self._path_for(tenant_id, doc_id)
        if path.exists():
            path.unlink()
