#!/usr/bin/env python3
"""Run one command on one SSH host and emit a JSON result."""

import argparse
import json
import subprocess


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def output_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH destination: [user@]hostname")
    parser.add_argument("--command", required=True, help="command interpreted by the remote shell")
    parser.add_argument("--port", type=int, help="SSH port")
    parser.add_argument("--connect-timeout", type=int, default=10, metavar="SECONDS")
    parser.add_argument("--timeout", type=float, help="overall command timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="print the SSH invocation without running it")
    args = parser.parse_args()

    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
    ]
    if args.port is not None:
        argv.extend(["-p", str(args.port)])
    argv.extend(["--", args.host, args.command])

    if args.dry_run:
        emit({"argv": argv, "command": args.command, "host": args.host, "status": "planned"})
        return 0

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        emit(
            {
                "command": args.command,
                "host": args.host,
                "returncode": 124,
                "status": "timeout",
                "stderr": output_text(error.stderr),
                "stdout": output_text(error.stdout),
            }
        )
        return 124
    except OSError as error:
        emit(
            {
                "command": args.command,
                "host": args.host,
                "returncode": 127,
                "status": "launch_error",
                "stderr": str(error),
                "stdout": "",
            }
        )
        return 127

    emit(
        {
            "command": args.command,
            "host": args.host,
            "returncode": result.returncode,
            "status": "completed",
            "stderr": result.stderr,
            "stdout": result.stdout,
        }
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
