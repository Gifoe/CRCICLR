from __future__ import annotations

import hashlib
import subprocess

import common


def main() -> None:
    head = common.git_head()
    relative = common.PROTOCOL_PATH.relative_to(common.REPO).as_posix()
    committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=common.REPO)
    disk = common.PROTOCOL_PATH.read_bytes()
    if committed.replace(b"\r\n", b"\n") != disk.replace(b"\r\n", b"\n"):
        raise RuntimeError("the current protocol is not byte-identical to the protocol-freeze commit")
    common.write_json(
        common.EXP / "PROTOCOL_FREEZE_COMMIT.json",
        {
            "pass": True,
            "protocol_freeze_commit": head,
            "protocol_sha256": hashlib.sha256(disk).hexdigest(),
            "recorded_before_new_setting_outcome_access": True,
        },
    )
    print(f"P4A_PROTOCOL_FREEZE_RECORDED {head}", flush=True)


if __name__ == "__main__":
    main()
