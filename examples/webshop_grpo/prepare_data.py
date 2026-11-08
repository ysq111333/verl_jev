"""Prepare all frozen train tasks and the first 100 real dev IDs for verl GRPO.

No WebShop/model imports and no test episodes. Parquet requires pyarrow in the
trainer environment. Dataset size is independent of the launcher's batch size:
use data.train_batch_size=16, rollout.n=8, total_training_steps=150.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from examples.webshop_grpo.webshop_env_service import read_manifest
except ModuleNotFoundError as error:
    if error.name != "examples":
        raise
    # Direct `python /absolute/path/prepare_data.py` invocation (no repo PYTHONPATH).
    from webshop_env_service import read_manifest


def make_rows(manifest, split, limit=None):
    if split not in {"train", "dev"}:
        raise ValueError("only train/dev datasets may be prepared")
    tasks = [task for task in manifest["tasks"] if task["split"] == split]
    if limit is not None:
        if limit <= 0 or len(tasks) < limit:
            raise ValueError(f"manifest does not contain {limit} {split} tasks")
        tasks = tasks[:limit]
    return [
        {
            "data_source": "webshop",
            "agent_name": "webshop",
            "prompt": [{"role": "user", "content": task["instruction"]}],
            "ability": "webshop",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "task_id": task["task_id"],
                "instruction": task["instruction"],
                "split": split,
                "manifest_id": manifest["manifest_id"],
                "index": index,
            },
        }
        for index, task in enumerate(tasks)
    ]


def prepare(manifest_path, output_dir, dev_size=100):
    manifest = read_manifest(manifest_path)
    train_rows, dev_rows = make_rows(manifest, "train"), make_rows(manifest, "dev", dev_size)
    if not train_rows:
        raise ValueError("manifest has no training tasks")
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        for suffix in (".parquet", ".parquet.tmp"):
            path = output_dir / f"{split}{suffix}"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite prepared data: {path}")
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        path = output_dir / f"{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(path)
    return {
        "manifest_id": manifest["manifest_id"],
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_path": str(output_dir / "train.parquet"),
        "dev_path": str(output_dir / "dev.parquet"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-size", type=int, default=100)
    args = parser.parse_args()
    import json

    print(json.dumps(prepare(args.manifest, args.output_dir, args.dev_size), sort_keys=True))


if __name__ == "__main__":
    main()
