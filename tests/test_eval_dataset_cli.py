import json

from eval.dataset_cli import add_from_trace, load_items, seed
from tests.eval.fake_backend import FakeEvalBackend


def test_load_items_parses_fields(tmp_path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps([
        {"id": "1", "question": "q1", "ground_truth": "a1",
         "relevant_chunk_ids": ["c1"], "tenant_id": "public"},
    ]))
    items = load_items(str(p))
    assert items[0].id == "1"
    assert items[0].expected_output == "a1"
    assert items[0].relevant_chunk_ids == ["c1"]


def test_seed_is_idempotent(tmp_path):
    be = FakeEvalBackend()
    items = load_items(str(_write(tmp_path)))
    assert seed(backend=be, dataset="ds", items=items) == 1
    seed(backend=be, dataset="ds", items=items)  # again
    assert len(be.get_dataset_items("ds")) == 1


def test_add_from_trace_records(tmp_path):
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    add_from_trace(backend=be, dataset="ds", trace_id="tr-1")
    assert be.trace_items["ds"] == ["tr-1"]


def _write(tmp_path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps([{"id": "1", "question": "q1", "ground_truth": "a1"}]))
    return p
