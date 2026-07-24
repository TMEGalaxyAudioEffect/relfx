#!/usr/bin/env python3
"""Fail on common internal paths, secrets, weights, audio, or oversized files."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
BLOCKED_SUFFIXES = {
    ".ckpt",
    ".flac",
    ".m4a",
    ".mp3",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
}
PATTERNS = {
    "absolute internal path": re.compile(r"/(?:data\\d*|home)/[^\\s\"']+"),
    "known internal identifier": re.compile(
        "faye" + "lliu|binglin_" + "share_data|quku_" + "top",
        re.IGNORECASE,
    ),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "generic assigned secret": re.compile(
        r"(?i)(?:api[_-]?key|password|passwd|secret|access[_-]?token)"
        r"\\s*[:=]\\s*[\"'][^\"']{8,}[\"']"
    ),
}


def main() -> None:
    findings = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append(f"{relative}: blocked artifact type")
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"{relative}: file exceeds 5 MiB")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "LICENSE":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {name}")

    if findings:
        print("\n".join(findings))
        raise SystemExit(f"Release scan failed with {len(findings)} finding(s)")
    print(f"Release scan passed: {ROOT}")


if __name__ == "__main__":
    main()
