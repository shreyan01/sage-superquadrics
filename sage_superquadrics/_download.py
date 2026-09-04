"""
Downloads and caches SAGE model weight files, the same way ultralytics'
YOLO auto-downloads a .pt file the first time you reference it by name.

Cache location follows the standard convention other ML libraries use
(torch.hub, ultralytics): ~/.cache/sage_superquadrics/ by default,
overridable via the SAGE_CACHE_DIR environment variable.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

# Real, hosted download URLs go here once a release exists to point at.
# Format: name -> (url, sha256_checksum). The checksum is NOT optional
# once a real URL is filled in -- silently trusting a downloaded file
# with no integrity check is exactly the kind of thing that goes wrong
# quietly, much later, for someone who isn't you.
KNOWN_MODELS = {
    "SAGE_V2": {
        "url": None,  # TODO: real hosted URL (GitHub Release asset) goes here once uploaded
        "sha256": "83d4a9cf4e43d7e838ae7b094135ecde911c8713363008eef4428aaf4955966a",  # verified against the real uploaded file
        "filename": "SAGE_V2.json",
    },
}


def _cache_dir():
    d = Path(os.environ.get("SAGE_CACHE_DIR", Path.home() / ".cache" / "sage_superquadrics"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model_path(name_or_path, progress=True):
    """Turns a model name (e.g. 'SAGE_V2') or an existing local file path
    into a real, local, ready-to-load file path -- downloading and
    caching it first if it's a known name not yet present locally.

    A literal path to an existing file is always used as-is, unchanged --
    this function only downloads for names it recognizes."""
    candidate = Path(name_or_path)
    if candidate.exists():
        return str(candidate)

    name = str(name_or_path)
    if name not in KNOWN_MODELS:
        raise FileNotFoundError(
            f"'{name_or_path}' is not an existing file and not a known model "
            f"name (known names: {list(KNOWN_MODELS.keys())}). If you meant a "
            f"local file, check the path; if you meant a named model, check "
            f"the spelling."
        )

    spec = KNOWN_MODELS[name]
    cache_path = _cache_dir() / spec["filename"]
    if cache_path.exists():
        if spec["sha256"] and _sha256(cache_path) != spec["sha256"]:
            print(f"Cached {cache_path} failed checksum verification -- re-downloading.")
        else:
            return str(cache_path)

    if spec["url"] is None:
        raise RuntimeError(
            f"'{name}' is a known model name but no download URL has been "
            f"configured yet for this release. Either wait for that to be "
            f"published, or pass a direct local file path to SAGE.load() "
            f"instead of the name '{name}'."
        )

    print(f"Downloading {name} to {cache_path} ...")

    def _report(block_num, block_size, total_size):
        if not progress or total_size <= 0:
            return
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(spec["url"], cache_path, reporthook=_report if progress else None)
    if progress:
        print()

    if spec["sha256"]:
        actual = _sha256(cache_path)
        if actual != spec["sha256"]:
            cache_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded {name} but its checksum doesn't match "
                f"(expected {spec['sha256']}, got {actual}). Deleted the "
                f"corrupted download -- try again, and if this keeps "
                f"happening, the hosted file itself may be the problem, "
                f"not your connection."
            )

    return str(cache_path)
