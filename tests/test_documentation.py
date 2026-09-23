import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "DATA_FORMAT.md",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "SCRIPT_INVENTORY.md",
]


def test_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in DOCS:
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.split("#", 1)[0]
            if target and "://" not in target and not (document.parent / target).exists():
                missing.append(f"{document.name}: {raw_target}")
    assert not missing, "missing documentation targets: " + ", ".join(missing)


def test_documented_brie_modules_exist() -> None:
    module_pattern = re.compile(r"(?:python -m |`)(brie\.[a-zA-Z0-9_.]+)")
    modules = {
        match.rstrip(".,`")
        for document in DOCS
        for match in module_pattern.findall(document.read_text(encoding="utf-8"))
    }
    missing = sorted(module for module in modules if importlib.util.find_spec(module) is None)
    assert not missing, "documented modules do not exist: " + ", ".join(missing)
