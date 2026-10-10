"""Peek into the big Steam player-count data and pull out just a few games.

Works on zip files OR already-extracted folders. Runs on YOUR computer.

Usage (zips or folders, any mix):
    python extract_steam_sample.py data/raw --apps 578080 252490 252950 271590 289070 --out steam_sample
    python extract_steam_sample.py PlayerCountHistoryPart1.zip PlayerCountHistoryPart2.zip --out steam_sample

Prints a short report (paste it back to Claude) and copies matching files to --out.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path


def appid_of(name):
    stem = Path(name).stem
    return stem if stem.isdigit() else re.sub(r"\D", "", stem)


def head(opener, n=6):
    with opener() as f:
        return [next(f, b"").decode("utf-8", "replace").rstrip() for _ in range(n)]


def entries(path):
    """Yield (name, size_bytes, opener) for every file in a zip or folder."""
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                yield str(f.relative_to(p)), f.stat().st_size, (lambda f=f: open(f, "rb"))
    else:
        z = zipfile.ZipFile(p)
        for i in z.infolist():
            if not i.is_dir():
                yield i.filename, i.file_size, (lambda i=i, z=z: z.open(i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="zip files and/or folders")
    ap.add_argument("--apps", nargs="*", default=["578080", "252490", "252950", "271590", "289070"])
    ap.add_argument("--out", default="steam_sample")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.apps)

    for src in args.sources:
        items = list(entries(src))
        total_mb = sum(s for _, s, _ in items) / 1e6
        print(f"\n=== {src}: {len(items)} files, {total_mb:,.0f} MB")
        print("first names:", [n for n, _, _ in items[:5]])

        hits = [(n, s, o) for n, s, o in items if appid_of(n) in wanted]
        print(f"matching requested appids: {len(hits)} -> {[appid_of(n) for n, _, _ in hits]}")

        sample = hits[0] if hits else (items[0] if items else None)
        if sample:
            n, s, o = sample
            print(f"\nfirst lines of {n} ({s/1e6:.1f} MB):")
            for line in head(o):
                print("   ", line)

        for n, s, o in hits:
            # keep the subfolder in the name so PlayerCountHistoryPart1/570.csv and
            # PriceHistory/570.csv can never overwrite each other
            safe = re.sub(r"[\\/]+", "__", n)
            target = out / f"{Path(src).stem}__{safe}"
            with o() as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
            print(f"copied {target.name} ({s/1e6:.1f} MB)")

    print(f"\nDone. Extracted files are in: {out.resolve()}")


if __name__ == "__main__":
    main()