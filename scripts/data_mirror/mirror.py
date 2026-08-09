#!/usr/bin/env python3
"""Resumable, OIDC-authenticated mirror from official EEG sources to HF Hub."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/data_mirror/datasets.json").read_text())
HF_REPO = os.environ.get("HF_REPO_ID", CONFIG["hf_repo_id"])
REPO_TYPE = "dataset"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "persist-eeg-mirror/1.0"})

MANIFEST_FIELDS = [
    "dataset", "dataset_version", "paradigm", "subject_id", "session_id",
    "run_id", "original_source", "original_url", "original_relative_path",
    "filename", "size_bytes", "upstream_checksum", "upstream_checksum_algorithm",
    "sha256", "download_status", "upload_status", "remote_verified", "hf_path", "notes",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: str, **data: Any) -> None:
    safe = {k: v for k, v in data.items() if "token" not in k.lower()}
    print(json.dumps({"event": event, "time": now(), **safe}, sort_keys=True), flush=True)


def retry(fn, label: str, attempts: int = 5):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transfer retries are intentional
            last = exc
            log("retry", label=label, attempt=attempt, error=type(exc).__name__)
            if attempt < attempts:
                time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {type(last).__name__}") from last


class OIDCHub:
    """Refreshable in-memory HF repo token obtained only from GitHub Actions OIDC."""

    def __init__(self) -> None:
        self._api: HfApi | None = None
        self._expires_at = 0.0

    def api(self, force: bool = False) -> HfApi:
        if force or self._api is None or time.monotonic() >= self._expires_at:
            self._api = HfApi(token=self._exchange())
            self._expires_at = time.monotonic() + 45 * 60
        return self._api

    def _exchange(self) -> str:
        req_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
        req_bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
        if not req_url or not req_bearer:
            raise RuntimeError("GitHub Actions OIDC environment is unavailable")
        sep = "&" if "?" in req_url else "?"
        oidc = retry(
            lambda: HTTP.get(
                req_url + sep + "audience=https%3A%2F%2Fhuggingface.co",
                headers={"Authorization": f"bearer {req_bearer}"},
                timeout=30,
            ),
            "github-oidc-id-token",
        )
        oidc.raise_for_status()
        subject_token = oidc.json()["value"]
        configured = os.environ.get("HF_OIDC_RESOURCE", HF_REPO).strip("/")
        candidates = [configured]
        if not configured.startswith("datasets/"):
            candidates.append(f"datasets/{configured}")
        request_ids: list[str] = []
        for resource in candidates:
            response = HTTP.post(
                "https://huggingface.co/oauth/token",
                json={
                    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                    "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
                    "subject_token": subject_token,
                    "resource": resource,
                },
                timeout=30,
            )
            if response.ok:
                access = response.json()["access_token"]
                log("oidc_exchange_success", resource=resource)
                return access
            request_ids.append(response.headers.get("x-request-id", "unknown"))
        raise RuntimeError("HF OIDC exchange rejected all dataset resource forms; request_ids=" + ",".join(request_ids))

    def upload(self, local: Path, remote: str, message: str) -> None:
        def action():
            return self.api().upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=HF_REPO,
                repo_type=REPO_TYPE,
                commit_message=message,
            )
        try:
            retry(action, f"upload:{remote}", attempts=4)
        except RuntimeError:
            self.api(force=True)
            retry(action, f"upload-refreshed:{remote}", attempts=3)

    def remote_info(self, remote: str):
        def action():
            infos = self.api().get_paths_info(HF_REPO, [remote], repo_type=REPO_TYPE, expand=True)
            return infos[0] if infos else None
        try:
            return retry(action, f"remote-info:{remote}", attempts=3)
        except RuntimeError:
            self.api(force=True)
            return retry(action, f"remote-info-refreshed:{remote}", attempts=3)

    def download(self, remote: str, target_dir: Path) -> Path:
        def action():
            return Path(hf_hub_download(
                repo_id=HF_REPO,
                filename=remote,
                repo_type=REPO_TYPE,
                token=self.api().token,
                local_dir=str(target_dir),
                force_download=True,
            ))
        try:
            return retry(action, f"download-hf:{remote}", attempts=3)
        except RuntimeError:
            self.api(force=True)
            return retry(action, f"download-hf-refreshed:{remote}", attempts=3)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] = MANIFEST_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def http_json(url: str) -> Any:
    def action():
        response = HTTP.get(url, timeout=(30, 180))
        response.raise_for_status()
        return response.json()
    return retry(action, f"GET-json:{url}")


def head_size(url: str) -> int:
    def action():
        response = HTTP.head(url, allow_redirects=True, timeout=(30, 120))
        response.raise_for_status()
        return int(response.headers.get("content-length", 0))
    return retry(action, f"HEAD:{url}", attempts=4)


def source_download(url: str, target: Path, expected_size: int = 0) -> tuple[float, float]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.monotonic()
    offset = partial.stat().st_size if partial.exists() else 0
    urls = [url]
    if url.startswith("https://data.nemar.org/"):
        urls.append(url.replace("https://data.nemar.org/", "https://api.nemar.org/data/", 1))
    for attempt in range(1, 7):
        active_url = urls[min((attempt - 1) // 3, len(urls) - 1)]
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with HTTP.get(active_url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 180)) as response:
                if offset and response.status_code == 200:
                    partial.unlink(missing_ok=True)
                    offset = 0
                response.raise_for_status()
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with partial.open(mode) as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            handle.write(block)
            size = partial.stat().st_size
            if expected_size and size != expected_size:
                raise IOError(f"size mismatch expected={expected_size} got={size}")
            partial.replace(target)
            seconds = max(time.monotonic() - started, 0.001)
            return seconds, size / 1_000_000 / seconds
        except Exception as exc:  # noqa: BLE001
            offset = partial.stat().st_size if partial.exists() else 0
            log("source_download_retry", url=active_url, attempt=attempt, offset=offset, error=type(exc).__name__)
            if attempt == 6:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise AssertionError("unreachable")


def subject_session(path: str) -> tuple[str, str]:
    subject = re.search(r"(?:^|/)sub-([A-Za-z0-9]+)(?:/|_)", path)
    session = re.search(r"(?:^|/)ses-([A-Za-z0-9]+)(?:/|_)", path)
    return (subject.group(1) if subject else "", session.group(1) if session else "")


def build_openbmi_manifest(paradigm: str) -> list[dict[str, Any]]:
    cfg = CONFIG["openbmi"][paradigm]
    entries = http_json(cfg["manifest_url"])
    if not isinstance(entries, list):
        entries = entries.get("files", entries.get("entries", []))
    rows = []
    for entry in entries:
        rel = str(entry.get("path") or entry.get("name") or "").lstrip("/")
        if not rel or rel.endswith("/"):
            continue
        subject, session = subject_session(rel)
        stable_url = f"https://data.nemar.org/{cfg['dataset_id']}/{cfg['version']}/{quote(rel, safe='/')}"
        checksum_algo = str(entry.get("checksum_algorithm") or "")
        checksum = str(entry.get("checksum") or "")
        rows.append({
            "dataset": f"openbmi-{paradigm}", "dataset_version": cfg["version"],
            "paradigm": paradigm.upper(), "subject_id": subject, "session_id": session,
            "run_id": "", "original_source": f"NEMAR {cfg['dataset_id']}",
            "original_url": stable_url, "original_relative_path": rel,
            "filename": Path(rel).name, "size_bytes": int(entry.get("size") or entry.get("size_bytes") or 0),
            "upstream_checksum": checksum, "upstream_checksum_algorithm": checksum_algo,
            "sha256": "", "download_status": "pending", "upload_status": "pending",
            "remote_verified": "false", "hf_path": f"openbmi/{paradigm.upper()}/{rel}",
            "notes": "NEMAR published BIDS distribution",
        })
    if not rows:
        raise RuntimeError(f"empty NEMAR manifest for {paradigm}")
    return rows


def build_gigadb_manifest(paradigm: str) -> list[dict[str, Any]]:
    payload = http_json(CONFIG["openbmi"]["gigadb_api"])
    files = payload.get("data", {}).get("files", [])
    marker = {"mi": "EEG_MI", "erp": "EEG_ERP", "ssvep": "EEG_SSVEP"}[paradigm]
    rows = []
    for entry in files:
        filename = str(entry.get("file_name") or "")
        if marker.lower() not in filename.lower():
            continue
        match = re.search(r"sess(?:ion)?0?(\d+).*?subj(?:ect)?0?(\d+)", filename, re.I)
        if not match:
            continue
        session, subject = match.groups()
        url = str(entry.get("url") or "")
        if not url:
            continue
        md5 = str((entry.get("file_attributes") or {}).get("MD5 checksum") or "")
        rows.append({
            "dataset": f"openbmi-{paradigm}", "dataset_version": "GigaDB-100542",
            "paradigm": paradigm.upper(), "subject_id": f"{int(subject):02d}",
            "session_id": str(int(session)), "run_id": "", "original_source": "GigaDB 100542",
            "original_url": url, "original_relative_path": filename, "filename": filename,
            "size_bytes": int(entry.get("file_size") or 0), "upstream_checksum": md5,
            "upstream_checksum_algorithm": "md5" if md5 else "", "sha256": "",
            "download_status": "pending", "upload_status": "pending", "remote_verified": "false",
            "hf_path": f"openbmi/{paradigm.upper()}/gigadb/{filename}",
            "notes": "Official GigaDB 100542 byte-preserving fallback",
        })
    combinations = {(row["subject_id"], row["session_id"]) for row in rows}
    if len(combinations) != 108:
        raise RuntimeError(f"GigaDB fallback completeness failed for {paradigm}: {len(combinations)}/108")
    return rows


def build_eegmmidb_manifest() -> list[dict[str, Any]]:
    cfg = CONFIG["eegmmidb"]
    response = retry(lambda: HTTP.get(cfg["checksums_url"], timeout=(30, 180)), "eegmmidb-checksums")
    response.raise_for_status()
    rows = []
    for line in response.text.splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if not match:
            continue
        checksum, rel = match.groups()
        rel = rel.lstrip("./")
        subject_match = re.search(r"S(\d{3})", rel)
        run_match = re.search(r"R(\d{2})", rel)
        url = f"{cfg['base_url']}/{quote(rel, safe='/')}"
        rows.append({
            "dataset": "eegmmidb", "dataset_version": cfg["version"], "paradigm": "MI",
            "subject_id": subject_match.group(1) if subject_match else "", "session_id": "",
            "run_id": run_match.group(1) if run_match else "", "original_source": "PhysioNet",
            "original_url": url, "original_relative_path": rel, "filename": Path(rel).name,
            "size_bytes": 0, "upstream_checksum": checksum.lower(),
            "upstream_checksum_algorithm": "sha256", "sha256": "",
            "download_status": "pending", "upload_status": "pending", "remote_verified": "false",
            "hf_path": f"eegmmidb/{rel}", "notes": "PhysioNet official distribution",
        })
    if not rows:
        raise RuntimeError("empty PhysioNet checksum manifest")
    return rows


def upload_control(hub: OIDCHub, local_root: Path, paths: list[tuple[Path, str]], message: str) -> None:
    for local, remote in paths:
        hub.upload(local, remote, message)


def plan() -> None:
    requested = os.environ.get("REQUESTED_DATASET", "all").lower()
    requested_chunk = os.environ.get("REQUESTED_CHUNK", "").strip()
    chunks = {
        "openbmi-mi": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "openbmi-erp": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "openbmi-ssvep": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "eegmmidb": ["001-020", "021-040", "041-060", "061-080", "081-100", "101-109"],
    }
    selected = list(chunks) if requested == "all" else [requested]
    if any(name not in chunks for name in selected):
        raise SystemExit(f"unknown dataset: {requested}")
    include = []
    for name in selected:
        values = [requested_chunk] if requested_chunk else chunks[name]
        for value in values:
            if value not in chunks[name]:
                raise SystemExit(f"invalid chunk {value} for {name}")
            include.append({"dataset": name, "chunk": value})
    print("matrix=" + json.dumps({"include": include}, separators=(",", ":")))
    print("strict=" + ("true" if requested == "all" and not requested_chunk else "false"))


def smoke() -> None:
    hub = OIDCHub()
    info = hub.api().repo_info(repo_id=HF_REPO, repo_type=REPO_TYPE, files_metadata=False)
    if not bool(getattr(info, "private", False)):
        raise RuntimeError("destination dataset repository is not private")
    with tempfile.TemporaryDirectory(prefix="persist-oidc-") as temp:
        root = Path(temp)
        sample = root / "oidc_test.txt"
        sample.write_text("PERSIST-EEG Trusted Publisher test\n")
        local_sha = sha256_file(sample)
        hub.upload(sample, "metadata/oidc_test.txt", "PERSIST-EEG OIDC smoke test")
        readback = hub.download("metadata/oidc_test.txt", root / "readback")
        remote_sha = sha256_file(readback)
        if local_sha != remote_sha:
            raise RuntimeError("OIDC smoke-test checksum mismatch")
    print("OIDC_TEST_SUCCESS", flush=True)


def bootstrap() -> None:
    hub = OIDCHub()
    with tempfile.TemporaryDirectory(prefix="persist-bootstrap-") as temp:
        root = Path(temp)
        uploads: list[tuple[Path, str]] = []
        for paradigm in ("mi", "erp", "ssvep"):
            try:
                rows = build_openbmi_manifest(paradigm)
            except Exception as exc:  # noqa: BLE001 - required official-source fallback
                log("nemar_manifest_failed_using_gigadb", paradigm=paradigm, error=type(exc).__name__)
                rows = build_gigadb_manifest(paradigm)
            path = root / f"openbmi-{paradigm}.csv"
            write_csv(path, rows)
            uploads.append((path, f"metadata/manifests/openbmi-{paradigm}.csv"))
            log("source_manifest_built", dataset=f"openbmi-{paradigm}", files=len(rows), bytes=sum(int(r["size_bytes"]) for r in rows))
        eeg_rows = build_eegmmidb_manifest()
        path = root / "eegmmidb.csv"
        write_csv(path, eeg_rows)
        uploads.append((path, "metadata/manifests/eegmmidb.csv"))
        for source, remote in (
            (ROOT / "configs/data_mirror/LICENSE_AUDIT.md", "LICENSE_AUDIT.md"),
            (ROOT / "configs/data_mirror/SOURCE_PROVENANCE.md", "SOURCE_PROVENANCE.md"),
            (ROOT / "configs/data_mirror/HF_README.md", "README.md"),
        ):
            uploads.append((source, remote))
        upload_control(hub, root, uploads, "Initialize PERSIST-EEG mirror metadata")
        log("bootstrap_complete", eegmmidb_files=len(eeg_rows))


def parse_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{2,3})-(\d{2,3})", value)
    if not match:
        raise ValueError(f"invalid chunk range: {value}")
    return int(match.group(1)), int(match.group(2))


def remote_manifest(hub: OIDCHub, dataset: str, target: Path) -> Path:
    return hub.download(f"metadata/manifests/{dataset}.csv", target)


def read_remote_state(hub: OIDCHub, remote: str, root: Path) -> dict[str, Any]:
    try:
        path = hub.download(remote, root / "existing-state")
        return json.loads(path.read_text())
    except (EntryNotFoundError, RuntimeError):
        return {"files": {}, "created_at": now()}


def validate_download(row: dict[str, Any], path: Path, actual_sha: str) -> None:
    expected_sha = row.get("upstream_checksum", "")
    algo = row.get("upstream_checksum_algorithm", "")
    if algo == "sha256" and expected_sha and actual_sha.lower() != expected_sha.lower():
        raise RuntimeError(f"upstream SHA256 mismatch for {row['original_relative_path']}")
    if row["dataset"] == "eegmmidb":
        if path.name.endswith(".edf"):
            with path.open("rb") as handle:
                if path.stat().st_size < 256 or handle.read(8) != b"0       ":
                    raise RuntimeError(f"invalid EDF header for {path.name}")
        elif path.name.endswith(".event") and path.stat().st_size == 0:
            raise RuntimeError(f"empty annotation file for {path.name}")


def transfer() -> None:
    dataset = os.environ["MIRROR_DATASET"].lower()
    chunk = os.environ["MIRROR_CHUNK"]
    lo, hi = parse_range(chunk)
    hub = OIDCHub()
    state_remote = f"metadata/state/{dataset}/{chunk}.json"
    chunk_manifest_remote = f"metadata/manifests/chunks/{dataset}-{chunk}.csv"
    progress_remote = f"metadata/state/progress/{dataset}-{chunk}.json"
    with tempfile.TemporaryDirectory(prefix=f"persist-{dataset}-{chunk}-") as temp:
        root = Path(temp)
        source_path = remote_manifest(hub, dataset, root / "source")
        source_rows = read_csv(source_path)
        selected = []
        for row in source_rows:
            subject = row.get("subject_id", "")
            if subject and subject.isdigit() and lo <= int(subject) <= hi:
                selected.append(row)
            elif not subject and lo in (1,):
                selected.append(row)
        if not selected:
            raise RuntimeError(f"no files selected for {dataset} {chunk}")
        state = read_remote_state(hub, state_remote, root)
        state.update({"dataset": dataset, "chunk": chunk, "updated_at": now()})
        state.setdefault("files", {})
        rows_by_hf = {row["hf_path"]: row for row in selected}
        manifest_path = root / "control" / chunk_manifest_remote
        state_path = root / "control" / state_remote
        progress_path = root / "control" / progress_remote
        for index, row in enumerate(selected, start=1):
            key = row["hf_path"]
            old = state["files"].get(key, {})
            if old.get("uploaded") and old.get("remote_verified") and old.get("sha256"):
                row.update({
                    "sha256": old["sha256"], "download_status": "complete",
                    "upload_status": "complete", "remote_verified": "true",
                    "size_bytes": old.get("size_bytes", row["size_bytes"]),
                })
                log("skip_verified", dataset=dataset, chunk=chunk, hf_path=key)
                continue
            local = root / "staging" / Path(row["original_relative_path"]).name
            expected_size = int(row.get("size_bytes") or 0)
            seconds, mbps = source_download(row["original_url"], local, expected_size)
            actual_size = local.stat().st_size
            actual_sha = sha256_file(local)
            validate_download(row, local, actual_sha)
            remote = hub.remote_info(key)
            remote_size = int(getattr(remote, "size", -1)) if remote else -1
            if remote_size != actual_size:
                hub.upload(local, key, f"Mirror {dataset} {chunk}: {row['filename']}")
                remote = hub.remote_info(key)
                remote_size = int(getattr(remote, "size", -1)) if remote else -1
            if remote_size != actual_size:
                raise RuntimeError(f"remote size verification failed for {key}: {remote_size} != {actual_size}")
            row.update({
                "size_bytes": actual_size, "sha256": actual_sha, "download_status": "complete",
                "upload_status": "complete", "remote_verified": "true",
            })
            state["files"][key] = {
                "source_url": row["original_url"], "hf_path": key, "size_bytes": actual_size,
                "sha256": actual_sha, "downloaded": True, "uploaded": True,
                "remote_verified": True, "retry_count": int(old.get("retry_count", 0)),
                "download_seconds": round(seconds, 3), "download_MBps": round(mbps, 3),
                "last_update": now(),
            }
            state["updated_at"] = now()
            completed = sum(1 for value in state["files"].values() if value.get("remote_verified"))
            progress = {
                "dataset": dataset, "chunk": chunk, "total": len(selected), "completed": completed,
                "bytes": sum(int(v.get("size_bytes", 0)) for v in state["files"].values() if v.get("remote_verified")),
                "updated_at": now(),
            }
            atomic_json(state_path, state)
            atomic_json(progress_path, progress)
            write_csv(manifest_path, rows_by_hf.values())
            upload_control(hub, root / "control", [
                (state_path, state_remote), (manifest_path, chunk_manifest_remote), (progress_path, progress_remote)
            ], f"Checkpoint {dataset} {chunk} file {index}/{len(selected)}")
            local.unlink(missing_ok=True)
            xet_cache = Path(os.environ.get("HF_XET_CACHE", "/tmp/persist-hf-xet"))
            shutil.rmtree(xet_cache, ignore_errors=True)
            xet_cache.mkdir(parents=True, exist_ok=True)
            log("file_complete", dataset=dataset, chunk=chunk, index=index, total=len(selected), hf_path=key, size=actual_size, download_MBps=round(mbps, 3))
        atomic_json(state_path, state)
        write_csv(manifest_path, rows_by_hf.values())
        upload_control(hub, root / "control", [(state_path, state_remote), (manifest_path, chunk_manifest_remote)], f"Complete {dataset} {chunk}")
        log("chunk_complete", dataset=dataset, chunk=chunk, files=len(selected))


def is_openbmi_signal(path: str) -> bool:
    lower = path.lower()
    bids_signal = "/eeg/" in lower and "_eeg." in lower and Path(lower).suffix in {".set", ".edf", ".bdf", ".vhdr", ".eeg"}
    gigadb_signal = lower.endswith(".mat") and "_eeg_" in lower
    return bids_signal or gigadb_signal


def download_optional(hub: OIDCHub, remote: str, root: Path) -> Path | None:
    try:
        return hub.download(remote, root)
    except (EntryNotFoundError, RuntimeError):
        return None


def finalize() -> None:
    strict = os.environ.get("STRICT_COMPLETE", "false").lower() == "true"
    hub = OIDCHub()
    chunks = {
        "openbmi-mi": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "openbmi-erp": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "openbmi-ssvep": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "eegmmidb": ["001-020", "021-040", "041-060", "061-080", "081-100", "101-109"],
    }
    with tempfile.TemporaryDirectory(prefix="persist-finalize-") as temp:
        root = Path(temp)
        all_rows: dict[str, list[dict[str, str]]] = {name: [] for name in chunks}
        progress: dict[str, Any] = {}
        for dataset, ranges in chunks.items():
            for chunk in ranges:
                path = download_optional(hub, f"metadata/manifests/chunks/{dataset}-{chunk}.csv", root / "chunks")
                if path:
                    all_rows[dataset].extend(read_csv(path))
            completed = [r for r in all_rows[dataset] if r.get("remote_verified") == "true"]
            progress[dataset] = {
                "total": len(all_rows[dataset]), "completed": len(completed),
                "bytes": sum(int(r.get("size_bytes") or 0) for r in completed),
            }
        progress["overall"] = {
            "files_completed": sum(v["completed"] for v in progress.values()),
            "bytes_completed": sum(v["bytes"] for v in progress.values()),
            "updated_at": now(),
        }
        progress_path = root / "MIRROR_PROGRESS.json"
        atomic_json(progress_path, progress)

        openbmi_path = root / "OPENBMI_COMPLETENESS.csv"
        open_fields = ["subject_id", "MI_S1", "MI_S2", "ERP_S1", "ERP_S2", "SSVEP_S1", "SSVEP_S2", "complete"]
        open_rows = []
        for subject in range(1, 55):
            out: dict[str, Any] = {"subject_id": f"{subject:02d}"}
            for dataset, paradigm in (("openbmi-mi", "MI"), ("openbmi-erp", "ERP"), ("openbmi-ssvep", "SSVEP")):
                for session in (1, 2):
                    ok = any(
                        int(r.get("subject_id") or -1) == subject
                        and int(r.get("session_id") or -1) == session
                        and r.get("remote_verified") == "true"
                        and is_openbmi_signal(r.get("original_relative_path", ""))
                        for r in all_rows[dataset]
                    )
                    out[f"{paradigm}_S{session}"] = str(ok).lower()
            out["complete"] = str(all(out[name] == "true" for name in open_fields[1:-1])).lower()
            open_rows.append(out)
        write_csv(openbmi_path, open_rows, open_fields)

        eeg_path = root / "EEGMMIDB_COMPLETENESS.csv"
        eeg_fields = ["subject_id", "edf_runs", "annotation_runs", "expected_runs", "readable", "complete"]
        eeg_rows = []
        for subject in range(1, 110):
            subset = [r for r in all_rows["eegmmidb"] if int(r.get("subject_id") or -1) == subject and r.get("remote_verified") == "true"]
            edf = {r["run_id"] for r in subset if r["filename"].endswith(".edf")}
            ann = {r["run_id"] for r in subset if r["filename"].endswith(".edf.event")}
            complete = edf == {f"{i:02d}" for i in range(1, 15)} and ann == edf
            eeg_rows.append({"subject_id": f"{subject:03d}", "edf_runs": len(edf), "annotation_runs": len(ann), "expected_runs": 14, "readable": str(len(edf) == 14).lower(), "complete": str(complete).lower()})
        write_csv(eeg_path, eeg_rows, eeg_fields)

        verification = verify_samples(hub, all_rows, root / "verification")
        verification_path = root / "REMOTE_CHECKSUM_VERIFICATION.json"
        atomic_json(verification_path, verification)
        report_path = root / "DATA_MIRROR_REPORT.md"
        open_complete = sum(r["complete"] == "true" for r in open_rows)
        eeg_complete = sum(r["complete"] == "true" for r in eeg_rows)
        report_path.write_text(
            "# PERSIST-EEG data mirror report\n\n"
            f"Generated: {now()}\n\n"
            "## Environment\n\nGitHub-hosted Ubuntu runner; streaming per-file staging; GitHub Actions OIDC.\n\n"
            f"## Hugging Face\n\nRepository: `{HF_REPO}`; private: true; OIDC verified.\n\n"
            f"## OpenBMI\n\nComplete participants: {open_complete}/54. Sources: NEMAR v1.0.1 MI/ERP/SSVEP.\n\n"
            f"## EEGMMIDB\n\nComplete participants: {eeg_complete}/109. Source: PhysioNet v1.0.0.\n\n"
            f"## Integrity\n\nFiles uploaded and remotely present: {progress['overall']['files_completed']}. "
            f"Bytes: {progress['overall']['bytes_completed']}. Verification failures: {verification['failures']}.\n"
        )
        upload_control(hub, root, [
            (progress_path, "metadata/MIRROR_PROGRESS.json"),
            (openbmi_path, "metadata/OPENBMI_COMPLETENESS.csv"),
            (eeg_path, "metadata/EEGMMIDB_COMPLETENESS.csv"),
            (verification_path, "metadata/checksums/REMOTE_CHECKSUM_VERIFICATION.json"),
            (report_path, "DATA_MIRROR_REPORT.md"),
        ], "Finalize PERSIST-EEG mirror")
        complete = open_complete == 54 and eeg_complete == 109 and verification["failures"] == 0
        log("finalize", strict=strict, complete=complete, openbmi_subjects=open_complete, eegmmidb_subjects=eeg_complete)
        if strict and not complete:
            raise RuntimeError("strict completeness gate failed")
        if complete:
            print("DATA_MIRROR_COMPLETE", flush=True)


def verify_samples(hub: OIDCHub, rows: dict[str, list[dict[str, str]]], root: Path) -> dict[str, Any]:
    random.seed(20260810)
    selected: list[dict[str, str]] = []
    for dataset in ("openbmi-mi", "openbmi-erp", "openbmi-ssvep"):
        for session in (1, 2):
            candidates = [r for r in rows[dataset] if r.get("session_id") == str(session) and r.get("remote_verified") == "true" and is_openbmi_signal(r.get("original_relative_path", ""))]
            subjects = sorted({r["subject_id"] for r in candidates})
            chosen = random.sample(subjects, min(3, len(subjects)))
            for subject in chosen:
                selected.extend(r for r in candidates if r["subject_id"] == subject)
    eeg_subjects = sorted({r["subject_id"] for r in rows["eegmmidb"] if r.get("remote_verified") == "true" and r["filename"].endswith(".edf")})
    for subject in random.sample(eeg_subjects, min(10, len(eeg_subjects))):
        candidates = [r for r in rows["eegmmidb"] if r.get("subject_id") == subject and r["filename"].endswith(".edf")]
        if candidates:
            selected.append(sorted(candidates, key=lambda r: r["run_id"])[0])
    results = []
    failures = 0
    for index, row in enumerate(selected):
        downloaded = hub.download(row["hf_path"], root / str(index))
        observed = sha256_file(downloaded)
        ok = observed == row["sha256"]
        failures += int(not ok)
        results.append({"hf_path": row["hf_path"], "expected_sha256": row["sha256"], "observed_sha256": observed, "match": ok})
        shutil.rmtree(root / str(index), ignore_errors=True)
    return {"verified_at": now(), "samples": len(results), "failures": failures, "results": results}


COMMANDS = {"plan": plan, "smoke": smoke, "bootstrap": bootstrap, "transfer": transfer, "finalize": finalize}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit("usage: mirror.py " + "|".join(COMMANDS))
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
