"""Download the thesaurus sources (README.md) into ``raw/``, verifying pinned SHA-256 digests.

    uv run python -I resources/thesaurus/fetch.py            # fetch what is missing, verify everything
    uv run python -I resources/thesaurus/fetch.py --verify   # verify only, download nothing

Each source lands in its own directory under ``raw/`` (gitignored). A download goes to a temporary file in that
directory and is moved into place only when its digest matches the pin; on a mismatch the temporary file is deleted
and the script exits 2. An existing file with the wrong digest is never overwritten silently: the script refuses
(exit 2) until it is removed by hand. The downloaded files are data and are never executed; ``build.py`` parses them.

Standard library only. Mirrors are tried in order; all must serve the same bytes (the pin decides).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
USER_AGENT = "hypanet-slop-thesaurus-fetch/1 (python urllib)"


@dataclass(frozen=True)
class Source:
    key: str  # directory under raw/
    filename: str
    sha256: str
    size: int
    urls: tuple[str, ...]
    licence: str


SOURCES: tuple[Source, ...] = (
    Source(
        key="oewn-2025",
        filename="english-wordnet-2025.xml.gz",
        sha256="9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe",
        size=11363503,
        urls=(
            "https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/"
            "english-wordnet-2025.xml.gz",
        ),
        licence="CC BY 4.0 (Open English WordNet), incorporating Princeton WordNet under the WordNet licence",
    ),
    Source(
        key="moby-thesaurus-ii",
        filename="mthesaur.txt",
        sha256="7c9742b1ed94435a893c0719b426725edb8a5242f8c526a75461bd6cee2dfd32",
        size=24853275,
        urls=(
            # Project Gutenberg eBook #3202; the PGLAF mirror first, as Gutenberg asks automated clients to use one.
            "https://aleph.pglaf.org/3/2/0/3202/files/mthesaur.txt",
            "https://www.gutenberg.org/files/3202/files/mthesaur.txt",
        ),
        licence="public domain (grant from the author, Grady Ward, January 2001)",
    ),
)


class FetchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, src: Source) -> None:
    """Raise FetchError unless ``path`` has the pinned size and digest."""
    size = path.stat().st_size
    if size != src.size:
        raise FetchError(f"{path}: {size} bytes, pinned {src.size}")
    got = sha256_file(path)
    if got != src.sha256:
        raise FetchError(f"{path}: sha256 {got}, pinned {src.sha256}")


def _download(url: str, dest_dir: Path, src: Source) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix=".part-", dir=dest_dir)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (pinned https)
            total = 0
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                total += len(chunk)
                if total > src.size:
                    raise FetchError(f"{url}: more than the pinned {src.size} bytes")
                out.write(chunk)
        verify(tmp, src)
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def fetch(src: Source, verify_only: bool = False) -> Path:
    dest_dir = RAW / src.key
    dest = dest_dir / src.filename
    if dest.exists():
        verify(dest, src)  # a wrong existing file is refused, never replaced
        return dest
    if verify_only:
        raise FetchError(f"{dest}: missing (run fetch.py without --verify)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in src.urls:
        if not url.startswith("https://"):
            raise FetchError(f"{url}: only https sources are pinned")
        print(f"fetch {src.key}: {url}", file=sys.stderr)
        try:
            tmp = _download(url, dest_dir, src)
        except FetchError as e:
            errors.append(str(e))
            continue
        except OSError as e:
            errors.append(f"{url}: {e}")
            continue
        tmp.chmod(0o644)
        tmp.replace(dest)
        return dest
    raise FetchError(f"{src.key}: every source failed:\n  " + "\n  ".join(errors))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="verify existing files only; download nothing")
    args = ap.parse_args(argv)
    status = 0
    for src in SOURCES:
        try:
            path = fetch(src, verify_only=args.verify)
        except FetchError as e:
            print(f"REFUSED {src.key}: {e}", file=sys.stderr)
            status = 2
            continue
        print(f"ok {src.key}: {path.relative_to(HERE)} sha256 {src.sha256}")
    return status


if __name__ == "__main__":
    sys.exit(main())
