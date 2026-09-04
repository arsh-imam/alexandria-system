#!/usr/bin/env python3
"""
ALEXANDRIA installer.

Brings a bare Linux machine to the frozen, evaluated system.

    python3 install.py /path/with/120GB/free

Downloads the 53 published objects, verifies every one against MANIFEST.tsv,
reassembles the split Wikipedia archive and verifies the whole, exposes the
tree at the path the frozen modules expect, and runs the verification gates.

Nothing is patched. The twelve runtime modules stay byte-identical to the ones
that were evaluated, which is what gate G0 checks.
"""

import argparse
import csv
import hashlib
import os
import platform
import shutil
import subprocess
import sys

REPO = "arshimam/alexandria-system"
MOUNT = "/media/pi/KINGSTON"
NEED_BYTES = 120 * 10**9
HERE = os.path.dirname(os.path.abspath(__file__))


def say(step, msg):
    print(f"\n[{step}] {msg}", flush=True)


def die(msg):
    sys.exit(f"\nFAILED: {msg}")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def preflight(target):
    say(1, "preflight")
    if sys.version_info < (3, 9):
        die(f"Python 3.9+ required, found {platform.python_version()}")
    arch = platform.machine()
    print(f"    python   {platform.python_version()}")
    print(f"    machine  {arch}")
    print(f"    system   {platform.system()}")
    if platform.system() != "Linux":
        die("Linux required (the frozen system uses libzim and llama.cpp on Linux)")
    if arch not in ("aarch64", "x86_64"):
        print(f"    note: untested architecture {arch}; proceeding")
    if arch != "aarch64":
        print("    note: gates G5 and G8 will SKIP - their reference values")
        print("          were recorded on aarch64. Everything else applies.")
    os.makedirs(target, exist_ok=True)
    st = os.statvfs(target)
    free = st.f_bavail * st.f_frsize
    print(f"    free     {free/1e9:.1f} GB at {target}")
    if free < NEED_BYTES:
        die(f"need {NEED_BYTES/1e9:.0f} GB free at {target}, found {free/1e9:.1f} GB")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        die("huggingface_hub missing. Run:\n"
            "    python3 -m pip install -r manifests/requirements-frozen.txt")


def install_deps():
    say(2, "installing pinned dependencies")
    req = os.path.join(HERE, "manifests", "requirements-frozen.txt")
    if not os.path.exists(req):
        die(f"missing {req}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", req])
    if r.returncode != 0:
        die("pip install failed")
    print("    ok")


def download(target):
    say(3, f"downloading {REPO}")
    from huggingface_hub import snapshot_download
    root = os.path.join(target, "local_ai")
    print("    ~96.7 GB, resumable - safe to interrupt and re-run")
    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=root,
                      max_workers=4)
    print(f"    downloaded to {root}")
    return root


def load_manifest(root):
    for p in (os.path.join(root, "MANIFEST.tsv"),
              os.path.join(HERE, "manifests", "distribution_manifest.tsv")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return list(csv.DictReader(f, delimiter="\t")), p
    die("no MANIFEST.tsv found")


def verify(root, rows):
    say(4, f"verifying {len(rows)} objects by SHA-256")
    bad, n = [], 0
    for i, r in enumerate(rows, 1):
        p = os.path.join(root, r["repo_path"])
        if not os.path.exists(p):
            bad.append(f"{r['repo_path']}: missing")
            continue
        if os.path.getsize(p) != int(r["size"]):
            bad.append(f"{r['repo_path']}: wrong size")
            continue
        got = sha256(p)
        if got != r["sha256"]:
            bad.append(f"{r['repo_path']}: SHA-256 mismatch")
        else:
            n += 1
        print(f"\r    {i}/{len(rows)} checked, {n} ok", end="", flush=True)
    print()
    if bad:
        for b in bad[:10]:
            print("    " + b)
        die(f"{len(bad)} object(s) failed verification. Re-run to re-download.")
    print(f"    all {n} objects verified")


def reassemble(root, rows):
    parts = sorted([r for r in rows if r["part_of"]],
                   key=lambda r: r["repo_path"])
    if not parts:
        return
    target_rel = parts[0]["part_of"]
    out = os.path.join(root, target_rel)
    want = parts[0]["whole_sha256"]
    say(5, f"reassembling {target_rel} from {len(parts)} parts")
    if os.path.exists(out) and sha256(out) == want:
        print("    already present and verified")
        for r in parts:
            p = os.path.join(root, r["repo_path"])
            if os.path.exists(p):
                os.remove(p)
        return
    h = hashlib.sha256()
    with open(out, "wb") as o:
        for r in parts:
            p = os.path.join(root, r["repo_path"])
            print(f"    appending {os.path.basename(p)}", flush=True)
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    o.write(b)
                    h.update(b)
            os.remove(p)          # free space as we go
    if h.hexdigest() != want:
        die(f"reassembled {target_rel} does not match:\n"
            f"  got  {h.hexdigest()}\n  want {want}")
    print(f"    verified {os.path.getsize(out)/1e9:.2f} GB, SHA-256 matches")


def expose(root):
    say(6, f"exposing the tree at {MOUNT}")
    src = os.path.dirname(root)
    if os.path.isdir(os.path.join(MOUNT, "local_ai")):
        print(f"    {MOUNT}/local_ai already present")
        return True
    parent = os.path.dirname(MOUNT)
    try:
        os.makedirs(parent, exist_ok=True)
        os.symlink(src, MOUNT)
        print(f"    symlinked {MOUNT} -> {src}")
        return True
    except OSError:
        pass
    print("    a symlink needs write access to /media. Run this once:")
    print(f"        sudo mkdir -p {MOUNT}")
    print(f"        sudo mount --bind {src} {MOUNT}")
    print(f"    to make it survive reboot, append to /etc/fstab:")
    print(f"        {src}  {MOUNT}  none  bind  0  0")
    print("    then re-run this installer to continue.")
    return False


def gates():
    say(7, "running the verification gates")
    v = os.path.join(HERE, "verify", "verify_alexandria.py")
    if not os.path.exists(v):
        print("    verify/verify_alexandria.py not found; skipping")
        return
    subprocess.run([sys.executable, v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="directory with at least 120 GB free")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    target = os.path.abspath(args.target)

    print("ALEXANDRIA installer")
    print(f"  dataset  {REPO}")
    print(f"  target   {target}")

    preflight(target)
    if not args.skip_deps and not args.verify_only:
        install_deps()
    root = (os.path.join(target, "local_ai") if args.verify_only
            else download(target))
    rows, mpath = load_manifest(root)
    print(f"    manifest: {mpath}")
    verify(root, rows)
    reassemble(root, rows)
    if expose(root):
        gates()
        print("\nDone. Start the system with:")
        print(f"    cd {HERE} && python3 src/main.py")
    else:
        print("\nStopped: finish the mount step above, then re-run with "
              "--verify-only to continue.")


if __name__ == "__main__":
    main()
