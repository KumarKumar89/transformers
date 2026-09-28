#!/usr/bin/env python3
"""Forensic snapshot tool for air-gapped test environments.

Collects an evidence bundle (env vars, network reachability, package versions,
git state, HF cache inventory) into a JSON report so that debugging results can
be reproduced / audited on an isolated machine.

Usage:
    python forensic_snapshot.py --output /tmp/forensic_report.json
"""

import argparse
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:  # noqa: BLE001 - forensic tools must never crash the audit
        return {"cmd": cmd, "returncode": None, "error": repr(exc)}


def collect_env():
    keys = [
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HOME",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_CACHE",
        "HF_ENDPOINT",
        "no_proxy",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ]
    return {k: os.environ.get(k) for k in keys if k in os.environ}


def collect_network(targets, port=443, timeout=2.0):
    """TCP-connect probe; on a truly air-gapped box every entry must fail."""
    results = {}
    for host in targets:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                results[host] = "REACHABLE"
        except Exception as exc:  # noqa: BLE001
            results[host] = f"UNREACHABLE ({type(exc).__name__}: {exc})"
    return results


def collect_packages():
    pkgs = ["transformers", "torch", "safetensors", "huggingface-hub", "datasets", "tokenizers", "numpy"]
    out = {}
    code = (
        "import json,sys\n"
        "from importlib.metadata import version, PackageNotFoundError\n"
        "names = sys.argv[1:]\n"
        "res = {}\n"
        "for n in names:\n"
        "    try:\n"
        "        res[n] = version(n)\n"
        "    except PackageNotFoundError:\n"
        "        res[n] = None\n"
        "print(json.dumps(res))\n"
    )
    p = subprocess.run([sys.executable, "-c", code, *pkgs], capture_output=True, text=True)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        out = {"error": p.stderr.strip()}
    return out


def collect_hf_cache():
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    inventory = {}
    for sub in ("hub", "hub/models--*", "xet"):
        root = os.path.join(hf_home, *sub.split("/"))
        if not os.path.exists(root):
            continue
        entries = []
        for dirpath, dirnames, filenames in os.walk(root):
            depth = os.path.relpath(dirpath, root).count(os.sep)
            if depth >= 2:
                dirnames[:] = []
            for f in filenames[:50]:
                fp = os.path.join(dirpath, f)
                try:
                    st = os.stat(fp)
                    entries.append({"path": os.path.relpath(fp, root), "size": st.st_size})
                except OSError:
                    pass
        inventory[sub] = {"root": root, "files": entries[:500]}
    return inventory


def collect_git(repo_dir):
    return {
        "head": _run(["git", "-C", repo_dir, "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _run(["git", "-C", repo_dir, "status", "--short"]),
        "dirty": bool(_run(["git", "-C", repo_dir, "status", "--porcelain"])["stdout"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/forensic_report.json")
    parser.add_argument(
        "--probe-hosts",
        nargs="*",
        default=["huggingface.co", "cdn-lfs.huggingface.co", "pypi.org", "github.com"],
    )
    parser.add_argument("--repo", default=os.getcwd())
    args = parser.parse_args()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "user": getpass.getuser(),
            "cwd": os.getcwd(),
        },
        "environment_variables": collect_env(),
        "network_probes": collect_network(args.probe_hosts),
        "air_gapped_verdict": all(v.startswith("UNREACHABLE") for v in collect_network(args.probe_hosts).values()),
        "packages": collect_packages(),
        "hf_cache_inventory": collect_hf_cache(),
        "git": collect_git(args.repo),
    }

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"Forensic report written to {args.output}")
    print(f"Air-gapped verdict: {report['air_gapped_verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
