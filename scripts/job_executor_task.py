#!/usr/bin/env python3
"""Submit one command through the existing AReaL-RSI JobExecutor."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from areal_rsi.api.execution import JobSpec
from areal_rsi.infra.execution.jobs import JobExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--cwd", default="/storage/lf02381138/attention-primitives")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    root = Path(os.environ.get("ATTENTION_JOB_ROOT", "/storage/lf02381138/attention-primitives-build/jobs"))
    root.mkdir(parents=True, exist_ok=True)
    spec = JobSpec(command=("/bin/bash", "-lc", args.command), cwd=args.cwd,
                   timeout_seconds=args.timeout)
    record = asyncio.run(JobExecutor(root).run(args.run_id, args.job_name, spec))
    print(json.dumps({"job_id": record.job_id, "state": record.state,
                      "exit_code": record.exit_code, "resources_released": record.resources_released,
                      "log_path": record.log_path}))


if __name__ == "__main__":
    main()
