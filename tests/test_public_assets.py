import json
from pathlib import Path

from sfcdi.discovery import manifest_content_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_all_public_json_files_parse():
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_public_manifest_hashes_are_self_consistent():
    for name in ("nist_ur5.json", "metropt3.json"):
        manifest = json.loads(
            (ROOT / "results" / "manifests" / name).read_text(encoding="utf-8")
        )
        assert manifest_content_sha256(manifest) == manifest["manifest_sha256"]


def test_public_files_do_not_contain_machine_specific_paths():
    excluded = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "build",
        "figures",
        "sf_cdi.egg-info",
    }
    forbidden_root = "/" + "root" + "/projects/"
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".pdf", ".npz", ".pt"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert forbidden_root not in text
        assert not any("\u4e00" <= character <= "\u9fff" for character in text)
