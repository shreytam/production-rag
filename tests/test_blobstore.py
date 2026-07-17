import pytest

from providers.blobstore.local_disk import LocalDiskBlobStore


def test_put_get_roundtrip(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path))
    store.put("tenant1/doc1.bin", b"payload")
    assert store.get("tenant1/doc1.bin") == b"payload"


def test_get_missing_raises(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path))
    with pytest.raises(KeyError):
        store.get("nope")


def test_delete_removes(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path))
    store.put("k", b"x")
    store.delete("k")
    with pytest.raises(KeyError):
        store.get("k")
