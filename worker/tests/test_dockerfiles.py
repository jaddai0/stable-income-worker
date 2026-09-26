"""Static checks that the images build from the repository root (build.yml context `.`).

No Docker daemon is needed: every non-stage COPY source must exist relative to the repo
root and be admitted by the Dockerfile's root-relative .dockerignore allow-list.
"""
import fnmatch
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _allowed(rel: str, rules: list[str]) -> bool:
    ok = False
    for rule in rules:
        neg = rule.startswith("!")
        pat = rule[1:] if neg else rule
        hit = fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, pat.replace("/**", "/*")) or (
            pat.endswith("/**") and rel.startswith(pat[:-2]))
        if hit:
            ok = neg
    return ok


@pytest.mark.parametrize("backend", ["pytorch", "tensorrt"])
def test_copy_sources_exist_and_pass_the_dockerignore(backend):
    dockerfile = ROOT / "worker" / f"Dockerfile.{backend}"
    rules = [l.strip() for l in (ROOT / "worker" / f"Dockerfile.{backend}.dockerignore").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    copies = re.findall(r"^COPY\s+(?!--from)(.+)$", dockerfile.read_text(), flags=re.M)
    assert copies, "no COPY lines found"
    for line in copies:
        *sources, _dest = line.split()
        for src in sources:
            path = ROOT / src
            assert path.exists(), f"{backend}: COPY source {src} missing relative to the repo root"
            files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
            admitted = [p for p in files if _allowed(str(p.relative_to(ROOT)), rules)]
            assert admitted, f"{backend}: dockerignore excludes everything under {src}"
    # Nothing secret or heavy may be admitted.
    for bad in ("env.env", ".env", "worker/.venv/x", "benchmarks/results/a.mp3", "gateway/.tokenizer/t5gemma.bin"):
        assert not _allowed(bad, rules), bad
