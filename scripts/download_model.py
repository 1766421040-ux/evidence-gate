"""Download only the pinned public Qwen snapshot; no remote code is executed."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/base-model")
    parser.add_argument("--weight-mirror", choices=("huggingface", "modelscope"), default="huggingface")
    args = parser.parse_args()
    root = Path(args.output)
    patterns = ["*.json", "*.txt", "*.model", "LICENSE", "README.md"]
    if args.weight_mirror == "huggingface":
        patterns.append("model.safetensors")
    snapshot_download(REPO, revision=REVISION, local_dir=root, allow_patterns=patterns)
    weight = root / "model.safetensors"
    if args.weight_mirror == "modelscope" and not weight.exists():
        # Metadata still comes from the pinned HF revision. The mirror weight
        # must have the identical published LFS SHA256 before it is accepted.
        url = f"https://modelscope.cn/models/{REPO}/resolve/master/model.safetensors"
        part = root / "model.safetensors.part"
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256(part) != WEIGHT_SHA256:
            raise ValueError("Mirror checksum mismatch; untrusted weight was not installed")
        part.replace(weight)
    if sha256(weight) != WEIGHT_SHA256:
        raise ValueError("Pinned model checksum mismatch")
    print(json.dumps({"model": REPO, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
                      "license": "Apache-2.0", "verified": True}, indent=2))


if __name__ == "__main__":
    main()
