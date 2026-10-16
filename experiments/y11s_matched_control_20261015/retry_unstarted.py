"""Archive the zero-epoch font failure after the user's explicit retry request."""
from pathlib import Path
import shutil
from run_y11s import HERE, FOLDER, read, write, sha, utc, process_alive, verify_inputs, launch_if_idle


def main():
    cfg = read(HERE / "protocol.json")
    state = read(HERE / "status.json")
    failure = read(HERE / "startup_failure.json")
    record = read(HERE / "training_record.json")
    assert state["status"] == "failed" and "Arial.ttf" in state["error"]
    assert not process_alive(state["pid"]), "Previous process is still alive."
    assert failure["completed_epochs"] == 0 and not failure["checkpoint_written"]
    assert not record.get("effective_arguments_verified"), "Training initialization had progressed; do not restart."
    assert record["protocol_sha256"] == sha(HERE / "protocol.json")
    assert [p.relative_to(FOLDER).as_posix() for p in FOLDER.rglob("*") if p.is_file()] == ["args.yaml"]
    verify_inputs(cfg)
    font_hash = "525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9"
    for relative in ("Ultralytics/Arial.ttf", "ultralytics/Ultralytics/Arial.ttf"):
        assert sha(HERE / relative) == font_hash
    archive = HERE / "startup_archive/font_download_failure_20261015_0744"
    assert not archive.exists()
    archive.mkdir(parents=True)
    entries = []
    for name in ("status.json", "process.json", "stdout.log", "stderr.log", "training_record.json"):
        source, target = HERE / name, archive / name
        assert source.resolve().is_relative_to(HERE.resolve()) and target.resolve().is_relative_to(HERE.resolve())
        entries.append({"source": str(source), "archive": str(target), "sha256": sha(source)})
        source.rename(target)
    shutil.copy2(HERE / "startup_failure.json", archive / "startup_failure.json")
    destination = archive / "runs" / FOLDER.name
    destination.parent.mkdir(parents=True)
    assert FOLDER.resolve().is_relative_to(HERE.resolve()) and destination.resolve().is_relative_to(HERE.resolve())
    FOLDER.rename(destination)
    write(archive / "manifest.json", entries)
    write(HERE / "retry_scope.json", {"authorized_at_utc": utc(), "user_request": "可以开始训练了",
        "prior_completed_epochs": 0, "prior_failure_archived_at": str(archive), "protocol_sha256": sha(HERE / "protocol.json"),
        "changes": "Use repaired local font cache; training configuration and data unchanged."})
    write(HERE / "status.json", {"status": "waiting_for_gpu", "updated_at_utc": utc(), "retry_authorized": True})
    launch_if_idle(cfg)


if __name__ == "__main__":
    main()
