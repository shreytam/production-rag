from __future__ import annotations

from pathlib import Path


class LocalDiskBlobStore:
    def __init__(self, root: str = ".cache/uploads") -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        root = self._root.resolve()
        p = (self._root / key).resolve()
        if p != root and not p.is_relative_to(root):
            raise KeyError(f"illegal blob key: {key}")  # path traversal guard
        return p

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise KeyError(key)
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
