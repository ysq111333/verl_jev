"""Verify a completed synchronous 8-rank checkpoint before formal launch."""
from pathlib import Path
import argparse
import json


def verify(root, step):
    root = Path(root).resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text().strip()) != step:
        raise ValueError("checkpoint completion tracker missing or wrong step")
    folder = root / f"global_step_{step}"
    required = [folder / "data.pt", folder / "actor/fsdp_config.json", folder / "actor/huggingface/config.json"]
    config = folder / "actor/fsdp_config.json"
    if not config.is_file() or json.loads(config.read_text()).get("world_size") != 8:
        raise ValueError("checkpoint FSDP world size is not 8")
    for rank in range(8):
        for kind in ("model", "optim", "extra_state"):
            required.append(folder / "actor" / f"{kind}_world_size_8_rank_{rank}.pt")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"incomplete checkpoint: {missing}")
    return {"step": step, "root": str(root), "files_checked": len(required), "status": "complete"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.step)))
