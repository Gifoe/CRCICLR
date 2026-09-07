#!/usr/bin/env python3
"""Publish a redacted Actions job status to a non-triggering repository path."""

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request(method: str, url: str, token: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "persist-eeg-status/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def redact(text: str) -> str:
    text = re.sub(r"hf_[A-Za-z0-9]{10,}", "[REDACTED]", text)
    text = re.sub(r"(?i)(authorization:\s*(?:bearer|token)\s+)\S+", r"\1[REDACTED]", text)
    return text[-12000:]


def main() -> None:
    token = os.environ["GITHUB_STATUS_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", os.environ["STATUS_NAME"])
    outcome = os.environ.get("STATUS_OUTCOME", "unknown")
    log_path = Path(os.environ.get("STATUS_LOG", ""))
    tail = redact(log_path.read_text(errors="replace")) if log_path.is_file() else "log file unavailable"
    status = {
        "name": name,
        "outcome": outcome,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "sha": os.environ.get("GITHUB_SHA"),
        "workflow": os.environ.get("GITHUB_WORKFLOW"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "log_tail": tail,
    }
    path = f"mirror_status/{name}.json"
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    encoded = base64.b64encode((json.dumps(status, indent=2) + "\n").encode()).decode()
    for attempt in range(5):
        sha = None
        try:
            existing = request("GET", api + "?ref=main", token)
            sha = existing.get("sha")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        payload = {
            "message": f"mirror status: {name} {outcome}",
            "content": encoded,
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha
        try:
            request("PUT", api, token, payload)
            print(f"STATUS_PUBLISHED {name} {outcome}")
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (409, 422) or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
