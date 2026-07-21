"""Dataset curation CLI: seed a dataset from a local file, or promote a trace.

`seed` bootstraps/updates a Langfuse dataset from a committed JSON file (also used
by tests). `add-from-trace` promotes a production trace to a dataset item; the
expected output is filled in later by a human in the Langfuse UI.
"""

from __future__ import annotations

import argparse
import json

from eval.langfuse_eval import EvalBackend, GoldenItem, build_backend


def load_items(path: str) -> list[GoldenItem]:
    with open(path) as f:
        raw = json.load(f)
    return [
        GoldenItem(
            id=str(r["id"]),
            question=r["question"],
            expected_output=r.get("ground_truth", r.get("expected_output", "")),
            relevant_chunk_ids=list(r.get("relevant_chunk_ids", [])),
            tenant_id=r.get("tenant_id", "public"),
        )
        for r in raw
    ]


def seed(*, backend: EvalBackend, dataset: str, items: list[GoldenItem]) -> int:
    backend.ensure_dataset(dataset)
    for item in items:
        backend.upsert_item(dataset=dataset, item=item)
    return len(items)


def add_from_trace(*, backend: EvalBackend, dataset: str, trace_id: str) -> None:
    backend.ensure_dataset(dataset)
    backend.add_item_from_trace(dataset=dataset, trace_id=trace_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Langfuse dataset curation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="Seed a dataset from a local JSON file.")
    p_seed.add_argument("--dataset", required=True)
    p_seed.add_argument("--items", required=True, help="Path to a JSON array of items.")

    p_trace = sub.add_parser("add-from-trace", help="Promote a trace to a dataset item.")
    p_trace.add_argument("--dataset", required=True)
    p_trace.add_argument("--trace-id", required=True)

    args = parser.parse_args()
    backend = build_backend()

    if args.command == "seed":
        count = seed(backend=backend, dataset=args.dataset,
                     items=load_items(args.items))
        print(f"[dataset] Seeded {count} items into '{args.dataset}'")
    elif args.command == "add-from-trace":
        add_from_trace(backend=backend, dataset=args.dataset, trace_id=args.trace_id)
        print(f"[dataset] Added item from trace '{args.trace_id}' to '{args.dataset}'")


if __name__ == "__main__":
    main()
