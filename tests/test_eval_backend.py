from eval.langfuse_eval import GoldenItem, RunItemScores
from tests.eval.fake_backend import FakeEvalBackend


def _item(i):
    return GoldenItem(id=i, question=f"q{i}", expected_output=f"a{i}",
                      relevant_chunk_ids=[f"c{i}"], tenant_id="public")


def test_seed_and_get_items_roundtrip():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    be.upsert_item(dataset="ds", item=_item("1"))
    be.upsert_item(dataset="ds", item=_item("1"))  # idempotent by id
    be.upsert_item(dataset="ds", item=_item("2"))
    items = be.get_dataset_items("ds")
    assert sorted(i.id for i in items) == ["1", "2"]
    assert items[0].question == "q1"


def test_run_experiment_executes_task_and_evaluators():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    be.upsert_item(dataset="ds", item=_item("1"))

    def task(item):
        return {"answer": item.question.upper()}

    def evaluator(item, output):
        return [("len", float(len(output["answer"])))]

    be.run_experiment(dataset="ds", run_name="r1",
                      items=be.get_dataset_items("ds"),
                      task=task, evaluators=[evaluator])
    scores = be.get_run_scores(dataset="ds", run_name="r1")
    assert scores == [RunItemScores(item_id="1", scores={"len": 2.0})]


def test_add_item_from_trace_records_source():
    be = FakeEvalBackend()
    be.ensure_dataset("ds")
    be.add_item_from_trace(dataset="ds", trace_id="tr-9")
    assert be.trace_items["ds"] == ["tr-9"]
