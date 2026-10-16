"""Vendor unchanged upstream U-Net modules at a pinned Git commit."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPO = "milesial/Pytorch-UNet"
TARGET = ROOT/"src/pcb_parser/vendor/pytorch_unet"


def fetch(url):
    request = Request(url, headers={"User-Agent": "pcb-parser-unet-import"})
    with urlopen(request, timeout=45) as response:
        return response.read()


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest_path = TARGET/"provenance.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            assert hashlib.sha256((TARGET/row["local"]).read_bytes()).hexdigest() == row["sha256"]
        print(json.dumps({"status": "already_imported_and_verified", "commit": manifest["commit"]}))
        return
    git = shutil.which("git") or str(Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe")
    commit = subprocess.check_output([git, "ls-remote", f"https://github.com/{REPO}.git", "refs/heads/master"],
                                     text=True, timeout=45).split()[0]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Invalid upstream commit")
    manifest = {"repository": f"https://github.com/{REPO}", "commit": commit,
                "license": "GPL-3.0", "modification": "upstream files unchanged; PCB adapter is separate", "files": []}
    for remote, local in [("unet/unet_parts.py", "unet_parts.py"),
                          ("unet/unet_model.py", "unet_model.py"), ("LICENSE", "LICENSE")]:
        payload = fetch(f"https://raw.githubusercontent.com/{REPO}/{commit}/{remote}")
        (TARGET/local).write_bytes(payload)
        manifest["files"].append({"remote": remote, "local": local, "sha256": hashlib.sha256(payload).hexdigest()})
    (TARGET/"__init__.py").write_text('"""Unchanged GPL-3.0 upstream modules; see provenance.json and LICENSE."""\n', encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "imported", "repository": manifest["repository"], "commit": commit}))


if __name__ == "__main__":
    main()
