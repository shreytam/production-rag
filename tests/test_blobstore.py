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


def test_parent_traversal_key_rejected(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path / "uploads"))
    with pytest.raises(KeyError):
        store.put("../evil/x", b"data")


def test_sibling_prefix_key_rejected(tmp_path):
    # root ".../uploads"; a sibling ".../uploads2evil" must NOT be accepted.
    store = LocalDiskBlobStore(root=str(tmp_path / "uploads"))
    with pytest.raises(KeyError):
        store.put("../uploads2evil/x", b"data")


def test_normal_nested_key_allowed(tmp_path):
    store = LocalDiskBlobStore(root=str(tmp_path / "uploads"))
    store.put("tenant1/doc1.bin", b"ok")
    assert store.get("tenant1/doc1.bin") == b"ok"
