"""Download the official AI-governance PDFs into data/corpus/.

Sources (verified November 2026):

* EU AI Act -- EUR-Lex, Regulation (EU) 2024/1689:
  https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689
* NIST AI Risk Management Framework 1.0 (NIST-AI-100-1):
  DOI https://doi.org/10.6028/NIST.AI.100-1 first, then the direct
  nvlpubs URL, then the 2025-08-29 web.archive.org copy as a fallback.
  (NIST restructured nvlpubs in 2026 and the direct link currently 404s;
  the DOI is the canonical locator.)
* OWASP Top 10 for LLM Applications 2025 -- the project's GitHub release:
  https://github.com/OWASP/www-project-top-10-for-large-language-model-applications/releases/download/2024/OWASP-Top-10-for-LLMs-v2025.pdf

Each download is checked (HTTP 200, Content-Type application/pdf, size
over 50 KB) and recorded in data/corpus/manifest.json with the URL that
actually worked, the SHA-256 and the download date. Nothing here needs a
key or a login. Run: python scripts/download_corpus.py [--out data/corpus]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

SOURCES: list[dict] = [
    {
        "doc_id": "EU-AI-ACT",
        "filename": "eu-ai-act.pdf",
        "urls": [
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
        ],
    },
    {
        "doc_id": "NIST-AI-RMF",
        "filename": "nist-ai-rmf.pdf",
        "urls": [
            "https://doi.org/10.6028/NIST.AI.100-1",
            "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
            "https://web-wp.archive.org/web/20250829034305/https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf",
        ],
    },
    {
        "doc_id": "OWASP-TOP10-LLM",
        "filename": "owasp-top10-llm-2025.pdf",
        "urls": [
            "https://github.com/OWASP/www-project-top-10-for-large-language-model-applications/releases/download/2024/OWASP-Top-10-for-LLMs-v2025.pdf",
        ],
    },
]

MIN_SIZE_BYTES = 50_000


def _download(url: str, dest: Path) -> tuple[bool, str]:
    """Download one URL; return (ok, message)."""
    request = urllib.request.Request(url, headers={"User-Agent": "rag-eval-harness/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except Exception as exc:  # network error, 404, timeout...
        return False, f"{type(exc).__name__}: {exc}"
    if "pdf" not in content_type.lower() and content_type.lower() not in (
        "application/octet-stream",  # GitHub release assets serve PDFs as this
    ):
        return False, f"unexpected Content-Type: {content_type}"
    if len(data) < MIN_SIZE_BYTES:
        return False, f"file too small ({len(data)} bytes); probably an error page"
    if not data.startswith(b"%PDF"):
        return False, "downloaded file does not start with %PDF"
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    return True, f"OK ({len(data):,} bytes, sha256 {sha[:12]}...)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/corpus")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents": [],
    }
    failures = 0
    for source in SOURCES:
        dest = out_dir / source["filename"]
        downloaded = False
        for url in source["urls"]:
            ok, message = _download(url, dest)
            print(f"{source['doc_id']}: {url}\n  -> {message}")
            if ok:
                manifest["documents"].append(
                    {
                        "doc_id": source["doc_id"],
                        "filename": source["filename"],
                        "url_used": url,
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                        "bytes": dest.stat().st_size,
                    }
                )
                downloaded = True
                break
        if not downloaded:
            print(f"{source['doc_id']}: FAILED -- all URLs exhausted.")
            failures += 1
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nManifest written to {out_dir / 'manifest.json'}")
    if failures:
        print(f"{failures} document(s) failed to download. See the notes in "
              "this script's docstring for the official pages to download "
              "them from by hand.")
        return 1
    print("All documents downloaded. Build the index with:")
    print("  python scripts/build_index.py --corpus data/corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
