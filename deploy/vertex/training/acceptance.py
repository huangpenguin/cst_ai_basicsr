"""Manually reproduce the approved eight-step Vertex training; never auto-submit."""

import argparse
import datetime
import fcntl
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs/deploy/training-readiness-20261118"
CLI = REPO / "deploy/vertex/cloud-cli.sh"
PROJECT = "hallowed-pipe-497802-i0"
REGION = "asia-northeast1"


def cloud(*args, source=None):
    env = os.environ.copy()
    if source:
        env["VERTEX_DATA_SOURCE"] = str(source)
    return subprocess.run(
        ["bash", str(CLI), *args, "--project=" + PROJECT, "--billing-project=" + PROJECT, "--format=json"],
        env=env,
        text=True,
        capture_output=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "status", "cancel", "fetch"])
    parser.add_argument("run_dir", type=Path, help="New absolute local directory for each manual run")
    args = parser.parse_args()
    directory = args.run_dir
    if not directory.is_absolute():
        parser.error("Use an absolute run directory")
    if args.action == "prepare":
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        spec = json.loads((EVIDENCE / "job-spec.json").read_text())
        run_id = "manual-acceptance-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = "gs://" + PROJECT + "-denoise-vertex/training-readiness/" + run_id
        spec["baseOutputDirectory"]["outputUriPrefix"] = prefix
        container_args = spec["workerPoolSpecs"][0]["containerSpec"]["args"]
        container_args[container_args.index("--output-uri") + 1] = prefix
        (directory / "job-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        (directory / "run.json").write_text(json.dumps({"display_name": run_id, "output_uri": prefix}, indent=2) + "\n")
        print("Prepared only. Review " + str(directory / "job-spec.json"))
        print("To submit: python3 " + str(Path(__file__).resolve()) + " submit " + str(directory))
        return
    if not directory.is_dir():
        parser.error("Prepare this run directory first")
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = json.loads((directory / "run.json").read_text())
        if args.action == "submit":
            spec = json.loads((directory / "job-spec.json").read_text())
            reference = json.loads((EVIDENCE / "job-spec.json").read_text())
            reference["baseOutputDirectory"]["outputUriPrefix"] = run["output_uri"]
            arguments = reference["workerPoolSpecs"][0]["containerSpec"]["args"]
            arguments[arguments.index("--output-uri") + 1] = run["output_uri"]
            if spec != reference:
                parser.error("The bounded acceptance configuration was modified; prepare a new run")
            marker = directory / "submission-attempted"
            if marker.exists():
                parser.error(
                    "Submission already attempted. Use status; if the response was lost, "
                    "find the display name in Vertex before any retry: " + run["display_name"]
                )
            marker.write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
            result = cloud(
                "ai",
                "custom-jobs",
                "create",
                "--region=" + REGION,
                "--display-name=" + run["display_name"],
                "--config=/input/data.tar",
                source=directory / "job-spec.json",
            )
            (directory / "submission.stderr").write_text(result.stderr)
            (directory / "job-created.json").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(
                    "Submission did not return success; inspect submission.stderr and Vertex jobs before retrying"
                )
            created = json.loads(result.stdout)
            print(created["name"])
            print(created.get("state"))
            return
        job = json.loads((directory / "job-created.json").read_text())["name"]
        if not re.fullmatch(r"projects/580558088344/locations/asia-northeast1/customJobs/[0-9]+", job):
            raise ValueError("Unexpected job resource")
        if args.action in ("status", "fetch"):
            result = cloud("ai", "custom-jobs", "describe", job, "--region=" + REGION)
            if result.returncode:
                raise RuntimeError(result.stderr)
            (directory / "job-latest.json").write_text(result.stdout)
            status = json.loads(result.stdout)
            print(status["state"])
            if args.action == "fetch":
                if status["state"] != "JOB_STATE_SUCCEEDED":
                    raise RuntimeError("Artifacts may be fetched as an accepted run only after cloud success")
                subprocess.run(
                    [
                        "bash",
                        str(REPO / "deploy/vertex/training/fetch-artifacts.sh"),
                        run["output_uri"],
                        str(directory / "artifacts"),
                    ],
                    check=True,
                )
        elif args.action == "cancel":
            result = cloud("ai", "custom-jobs", "cancel", job, "--region=" + REGION)
            print(result.stdout)
            if result.returncode:
                raise RuntimeError(result.stderr)


if __name__ == "__main__":
    main()
