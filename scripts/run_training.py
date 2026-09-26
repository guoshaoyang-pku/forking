#!/usr/bin/env python3
"""Plan or launch one explicit baseline/control run without overwriting outputs."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from validate_data import sha256, validate

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def result_check(run_dir: Path, run_id: str, config: dict) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text().splitlines() if line]
    require(summary["run_id"] == run_id, "run ID mismatch")
    require(summary["seed"] == config["seed"], "seed mismatch")
    require(summary["steps"] == config["steps"], "step count mismatch")
    steps = [row["step"] for row in rows]
    expected_steps = list(range(config["val_interval"], config["steps"] + 1, config["val_interval"]))
    require(steps == expected_steps, "missing or duplicate validation steps")
    for row in rows:
        require(all(math.isfinite(row[key]) for key in ("train_loss", "val_loss", "gap")), "non-finite logged loss")
        require(math.isclose(row["gap"], row["val_loss"] - row["train_loss"], abs_tol=1e-10), "logged gap mismatch")
    for key in ("final_train_loss", "final_val_loss", "final_gap"):
        require(math.isfinite(summary[key]), f"non-finite {key}")
    require(math.isclose(summary["final_gap"], summary["final_val_loss"] - summary["final_train_loss"], abs_tol=1e-10), "gap mismatch")
    for name in ("train_loss", "val_loss", "gap"):
        require(math.isclose(summary["final_" + name], rows[-1][name], abs_tol=1e-10), f"summary/log mismatch: {name}")
    for filename in ("freq_bin_loss.jsonl", "exact_freq_loss.jsonl"):
        diagnostics = [json.loads(line) for line in (run_dir / filename).read_text().splitlines() if line]
        require([row["step"] for row in diagnostics] == expected_steps, f"incomplete {filename}")
    return {"ok": True, "steps": summary["steps"], "final_gap": summary["final_gap"],
            "scope": "Run/artifact integrity. A particular scientific outcome is not a pass condition."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.json")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--freq-index", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Unique name, without the _fixed suffix")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--gpu", default="0", help="One physical GPU index or GPU UUID")
    parser.add_argument("--execute", action="store_true", help="Validate inputs and start GPU training; otherwise only print the command")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id) or args.run_id.endswith("_fixed"):
        parser.error("run-id must be a simple unique name without the _fixed suffix")
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", args.gpu):
        parser.error("gpu must identify one GPU index or GPU UUID")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())["trainer_args"]
    for reserved in ("run_id", "data_dir", "freq_index", "out_dir"):
        if reserved in config:
            parser.error(f"{reserved} is a launch argument, not a trainer_args entry")
    if config["steps"] < 1 or config["steps"] % config["val_interval"]:
        parser.error("steps must be positive and end on a validation step")
    if config["val_steps"]:
        parser.error("this minimal launcher uses periodic validation; val_steps must be empty")
    if config["total_batch_size"] % (config["device_batch_size"] * config["sequence_len"]):
        parser.error("total_batch_size must be divisible by device_batch_size * sequence_len")
    if config["train_shards"] != "1" or config["val_shards"] != "2,3,4,5,6,7,8,9,10,6542":
        parser.error("this minimal launcher validates the packaged baseline split only")
    run_id = args.run_id + "_fixed"
    run_dir = args.out_dir.resolve() / run_id
    if run_dir.exists():
        parser.error(f"refusing to overwrite existing output: {run_dir}")
    command = [sys.executable, "-u", str(ROOT / "code" / "train.py")]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + key)
        else:
            command.extend(["--" + key, str(value)])
    command.extend(["--run_id", run_id, "--data_dir", str(args.data_dir.resolve()),
                    "--freq_index", str(args.freq_index.resolve()), "--out_dir", str(args.out_dir.resolve())])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["NGLAB_ASYNC_DIAG"] = "1"
    print("CUDA_VISIBLE_DEVICES=" + shlex.quote(args.gpu) + " NGLAB_ASYNC_DIAG=1 " + shlex.join(command), flush=True)
    if not args.execute:
        print("Plan only: no inputs have been validated and no training has started. Add --execute to run.")
        return 0
    if not (ROOT / "code" / "train.py").is_file():
        parser.error("missing code/train.py")
    data_report = validate(args.data_dir.resolve(), args.freq_index.resolve())
    if not data_report["ok"]:
        print(json.dumps(data_report, indent=2), file=sys.stderr)
        return 1
    # Check in the same interpreter and CUDA visibility as the actual process.
    probe_code = "\n".join([
        "import json, sys, torch, numpy",
        "if not torch.cuda.is_available(): raise SystemExit('CUDA GPU required')",
        "if not torch.cuda.is_bf16_supported(): raise SystemExit('BF16 support required')",
        "print(json.dumps({'python': sys.version, 'torch': torch.__version__, 'numpy': numpy.__version__, "
        "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}))",
    ])
    probe = subprocess.run([sys.executable, "-c", probe_code],
        env=env, capture_output=True, text=True)
    if probe.returncode:
        print(probe.stderr, file=sys.stderr)
        return probe.returncode
    runtime = json.loads(probe.stdout)
    run_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command, "config": config, "config_sha256": sha256(config_path),
        "runtime": runtime, "inputs": data_report,
        "code_sha256": {p.name: sha256(p) for p in sorted((ROOT / "code").glob("*.py"))},
    }
    (run_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    with (run_dir / "train.log").open("x") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        print(f"Training exited with status {completed.returncode}; retained log: {run_dir / 'train.log'}", file=sys.stderr)
        return completed.returncode
    try:
        report = result_check(run_dir, run_id, config)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"Output verification failed: {exc}. Artifacts retained at {run_dir}", file=sys.stderr)
        return 1
    (run_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed and checked: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
