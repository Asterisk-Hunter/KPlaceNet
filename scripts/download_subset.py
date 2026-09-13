"""Subset downloader — dummy Flickr smoke + SAFE OSV-5M subset + IM2GPS3k note.

SAFETY (OSV-5M):
    The full OSV-5M repo (osv5m/osv5m) is ~259GB. This script NEVER calls
    snapshot_download or datasets.load_dataset. It downloads ONLY:
      - one metadata file (test.csv ~116MB for --split test), and
      - explicitly listed shard zips (images/train/NN.zip, images/test/NN.zip),
    then extracts ONLY the selected image ids.
    The full train.csv (~2.92GB) is NEVER fully downloaded unless the user
    passes the explicit opt-in --allow-full-train-csv. Otherwise train rows
    are found by chunked HTTP Range streaming with a byte cap
    (--csv-scan-cap-mb); if the cap is hit the script aborts with a clear
    error instead of silently downloading huge files.

Windows-safe: uses pathlib, no hardcoded absolute paths, ASCII-only prints.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence


OSV_REPO = "osv5m/osv5m"
OSV_N_TRAIN_SHARDS = 98  # images/train/00.zip .. 97.zip
OSV_N_TEST_SHARDS = 5  # images/test/00.zip .. 04.zip
OSV_TEST_CSV_MB = 116
OSV_TRAIN_CSV_GB = 2.92

DATASETS = {
    "flickr_geo_tiny": {
        "note": "Flickr-Geo debug subset (iteration, OOM checks). Real HF id unpublished; use --dry-run dummy CSV.",
    },
    "osv5m": {
        "hf_id": OSV_REPO,
        "note": "OSV-5M full is ~259GB / 5.1M images. This script fetches a small SAFE subset only (default 10000, train split).",
    },
    "im2gps3k": {
        "note": "IM2GPS3k - 3k eval only, never train (plan Sec 2). Download manually from the authors and place under data/im2gps3k/.",
    },
}

# CSV schema detection (OSV CSVs are not assumed; fail clearly if missing).
ID_CANDS = ("id", "image_id", "img_id", "imageid", "name", "key")
PATH_CANDS = ("image_path", "filepath", "file_path", "path", "file", "image", "filename")
LAT_CANDS = ("lat", "latitude")
LON_CANDS = ("lon", "lng", "long", "longitude")
EXTRA_CANDS = ("country", "region", "city", "year", "month")


class OSVDownloadError(RuntimeError):
    """Raised when a safe OSV download cannot proceed without huge fetches."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KPlaceNet safe subset downloader (no snapshot_download, no full-repo fetch)")
    p.add_argument(
        "--dataset",
        type=str,
        default="flickr_geo_tiny",
        choices=["flickr_geo_tiny", "osv5m", "im2gps3k"],
        help="Which dataset to fetch",
    )
    p.add_argument("--output-dir", type=str, default="data/flickr_geo_tiny", help="Output directory (relative or absolute)")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples to keep (OSV default: 10000)")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"],
                   help="OSV split to use. train = training subset (never split into eval). test = official OSV test, eval-only.")
    p.add_argument("--shards", type=str, default=None,
                   help="Comma-separated OSV shard numbers, e.g. '00,01,02'. If omitted, shards are discovered sequentially from 00.")
    p.add_argument("--max-shards", type=int, default=2,
                   help="Max shard zips to download for train split (test split: up to 5). Bounds multi-GB risk.")
    p.add_argument("--plan-only", action="store_true", help="Print the exact download plan and exit. No network, no files.")
    p.add_argument("--dry-run", action="store_true",
                   help="flickr: create dummy metadata.csv. osv5m: same as --plan-only (never fakes OSV data).")
    p.add_argument("--yes", action="store_true", help="Confirm multi-GB download. Without it the script prints the plan and exits before downloading.")
    p.add_argument("--allow-full-train-csv", action="store_true",
                   help="Explicit opt-in to download the full train.csv (~2.92GB) for exact sampling. Without it, train rows are found by capped streaming scan.")
    p.add_argument("--csv-scan-cap-mb", type=int, default=512, help="Max bytes of train.csv to stream-scan before aborting.")
    p.add_argument("--chunk-mb", type=int, default=8, help="HTTP Range chunk size (MB) for streaming train.csv.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    p.add_argument("--hf-token", type=str, default=None, help="Optional HuggingFace token (or set HF_TOKEN env) for rate limits/resume.")
    return p.parse_args()


def _token(args: argparse.Namespace) -> str | None:
    return args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


# ---------------------------------------------------------------------------
# Dummy CSV (flickr smoke only — never for OSV)
# ---------------------------------------------------------------------------


def make_dummy_csv(output_dir: Path, n: int = 50, seed: int = 42) -> Path:
    """Create a dummy metadata.csv with n rows (image_path,lat,lon).

    Also creates placeholder image files so dataset.py can open them
    without needing real downloads - for smoke overfit of 50 images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    try:
        from PIL import Image as _PILImage

        has_pil = True
    except ImportError:
        _PILImage = None
        has_pil = False

    rows = []
    for i in range(n):
        lat = rng.uniform(-60, 70)
        lon = rng.uniform(-180, 180)
        img_name = f"dummy_{i:05d}.jpg"
        img_path = images_dir / img_name

        if _PILImage is not None and not img_path.exists():
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            _PILImage.new("RGB", (64, 64), color).save(img_path, "JPEG")

        rel = Path("images") / img_name
        rows.append((rel.as_posix(), f"{lat:.6f}", f"{lon:.6f}"))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "lat", "lon"])
        w.writerows(rows)

    if not has_pil:
        for rel, _, _ in rows:
            p = output_dir / rel
            if not p.exists():
                p.write_bytes(b"")  # will fail on open - user needs PIL; warn
        print("[warn] Pillow not installed - dummy images are empty. Install Pillow before smoke train.")

    print(f"[dry-run] wrote {n} rows -> {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# OSV safe helpers (hf_hub_download for single files only — never snapshots)
# ---------------------------------------------------------------------------


def _osv_shard_name(split: str, shard: str) -> str:
    return f"images/{split}/{shard}.zip"


def _parse_shards(spec: str | None, split: str, max_shards: int) -> list[str]:
    """Normalize --shards spec to zero-padded ids, or sequential discovery."""
    n_all = OSV_N_TRAIN_SHARDS if split == "train" else OSV_N_TEST_SHARDS
    if spec:
        out = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if not tok.isdigit():
                raise OSVDownloadError(f"Bad --shards entry {tok!r}: expected numbers like '00,01,02'.")
            idx = int(tok)
            if not (0 <= idx < n_all):
                raise OSVDownloadError(f"Shard {tok!r} out of range for split={split} (0..{n_all - 1}).")
            out.append(f"{idx:02d}")
        if not out:
            raise OSVDownloadError("Empty --shards spec.")
        # De-dup, keep order.
        seen = set()
        dedup = [s for s in out if not (s in seen or seen.add(s))]
        if len(dedup) > max_shards:
            print(f"[note] --shards listed {len(dedup)} shards; truncated to --max-shards={max_shards}.")
            return dedup[:max_shards]
        return dedup
    # Sequential discovery from 00.
    n = min(max_shards, n_all)
    return [f"{i:02d}" for i in range(n)]


def _hub_url(filename: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(OSV_REPO, filename, repo_type="dataset")


def _hub_download(filename: str, local_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=OSV_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
    )


def _head_size(url: str, token: str | None) -> int | None:
    """HEAD request size in bytes, or None if unknown. Network: download path only."""
    try:
        headers = {"Accept-Encoding": "identity"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "unknown size"
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n} B"


def _find_col(columns: Sequence[str], *cands: str) -> str | None:
    norm = {c.lower().strip(): c for c in columns}
    for cand in cands:  # exact match first
        if cand in norm:
            return norm[cand]
    for c in columns:  # substring fallback
        lc = c.lower()
        for cand in cands:
            if cand in lc:
                return c
    return None


def _stem(name: str) -> str:
    return Path(name).stem


def _stream_match_train_csv(
    url: str,
    wanted: dict[str, dict],
    max_samples: int,
    chunk_bytes: int,
    cap_bytes: int,
    token: str | None,
) -> tuple[dict[str, dict], int, bool]:
    """Stream remote train.csv in Range chunks; keep rows whose id matches wanted.

    Returns (found, bytes_scanned, eof_reached). Stops early once max_samples
    matched. Raises OSVDownloadError if the scan cap is exceeded first.
    """
    found: dict[str, dict] = {}
    header: list[str] | None = None
    id_col = path_col = lat_col = lon_col = None
    extra_cols: list[str] = []
    start = 0
    carry = ""
    first = True
    bytes_scanned = 0
    eof = False

    while True:
        if bytes_scanned + chunk_bytes > cap_bytes and not found:
            pass  # cap enforced below after accounting this chunk
        end = start + chunk_bytes - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                raw = resp.read(chunk_bytes + 1024)
        except Exception as e:
            raise OSVDownloadError(f"Range fetch failed at byte {start}: {e}. "
                                   "Check repository availability and optional token/rate-limit settings.")
        if not raw:
            eof = True
            break
        bytes_scanned += len(raw)
        text = raw.decode("utf-8-sig" if first else "utf-8", errors="replace")
        buf = carry + text
        # If server ignored Range (200 + full body), only use what we read.
        lines = buf.split("\n")
        if len(raw) >= chunk_bytes + 1024 and not first:
            # Possibly truncated mid-stream; treat last line as carry anyway.
            pass
        carry = lines.pop()  # incomplete trailing line -> next chunk
        if first:
            first = False
            if not lines:
                raise OSVDownloadError("Could not read CSV header in first chunk.")
            header = next(csv.reader([lines.pop(0)]))
            if not header:
                raise OSVDownloadError("Empty CSV header.")
            id_col = _find_col(header, *ID_CANDS)
            path_col = _find_col(header, *PATH_CANDS)
            lat_col = _find_col(header, *LAT_CANDS)
            lon_col = _find_col(header, *LON_CANDS)
            if lat_col is None or lon_col is None:
                raise OSVDownloadError(f"train.csv header lacks lat/lon columns (got: {header}).")
            if id_col is None and path_col is None:
                raise OSVDownloadError(f"train.csv header has neither id nor path column (got: {header}).")
            extra_cols = [c for c in header if _find_col([c], *EXTRA_CANDS) is not None]
        for row in csv.DictReader(lines, fieldnames=header):
            keys = set()
            if id_col and row.get(id_col):
                keys.add(str(row[id_col]).strip())
            if path_col and row.get(path_col):
                keys.add(_stem(str(row[path_col]).strip()))
            hit = next((k for k in keys if k in wanted), None)
            if hit is not None and hit not in found:
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                except (TypeError, ValueError):
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                rec: dict[str, Any] = {"lat": lat, "lon": lon, "id": hit, "split": "train"}
                for c in extra_cols:
                    rec[c] = row.get(c, "")
                if path_col and row.get(path_col):
                    rec["_src_path"] = str(row[path_col]).strip()
                found[hit] = rec
                if len(found) >= max_samples:
                    return found, bytes_scanned, False
        if len(raw) < chunk_bytes:
            eof = True
            break
        if bytes_scanned >= cap_bytes:
            break
        start += chunk_bytes
        _ = status  # (206 expected; 200 tolerated — we only consume chunk windows)

    if carry.strip() and header is not None:
        pass  # trailing partial line at EOF/cap is negligible; ignore honestly
    return found, bytes_scanned, eof


def _extract_members(zip_path: Path, members: list[str], dest_dir: Path) -> tuple[int, list[str]]:
    """Extract only the named members. Returns (extracted, missing)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        by_stem = {}
        for nm in names:
            if nm.endswith("/"):
                continue
            by_stem.setdefault(_stem(nm), nm)
        for m in members:
            src = by_stem.get(m, m if m in names else None)
            if src is None:
                missing.append(m)
                continue
            data = zf.read(src)
            ext = Path(src).suffix or ".jpg"
            out = dest_dir / f"{m}{ext}"
            out.write_bytes(data)
            n += 1
    return n, missing


# ---------------------------------------------------------------------------
# OSV plan +実行
# ---------------------------------------------------------------------------


def print_osv_plan(args: argparse.Namespace, output_dir: Path) -> None:
    max_samples = args.max_samples or 10000
    shards = _parse_shards(args.shards, args.split, args.max_shards)
    print("[plan] OSV-5M safe-subset plan (NO network performed, nothing downloaded)")
    print(f"  repo: {OSV_REPO} (public; token optional for rate limits/resume)")
    print(f"  split: {args.split} | rows requested: {max_samples} (seed {args.seed})")
    if args.split == "test":
        print(f"  metadata: test.csv (~{OSV_TEST_CSV_MB}MB) via single-file hf_hub_download (exact file, stated size)")
        print(f"  shards: images/test/{{{' '.join(shards)}}}.zip - {len(shards)} file(s), sizes checked via HEAD at download time")
    else:
        if args.allow_full_train_csv:
            print(f"  metadata: FULL train.csv (~{OSV_TRAIN_CSV_GB}GB) via single-file hf_hub_download (explicit opt-in given)")
        else:
            print(f"  metadata: train.csv (~{OSV_TRAIN_CSV_GB}GB full) NEVER fully downloaded; chunked Range streaming scan, "
                  f"cap {args.csv_scan_cap_mb}MB, chunk {args.chunk_mb}MB")
        print(f"  shards: sequential {args.split} shards [{', '.join(shards)}] (up to --max-shards={args.max_shards}); "
              f"sizes checked via HEAD at download time")
    print(f"  output: {output_dir / 'metadata.csv'} with image_path,lat,lon,id,split (+country/region if source has them)")
    print(f"  provenance: split={args.split} for every row. Train is NEVER split into eval.")
    print(f"  eval: use --split test (official OSV test, eval-only) into a separate output dir; never train on it.")
    print("  honesty: final row count is the truth - the script never claims '10k' if fewer were extracted.")
    if not args.yes:
        print("  gate: re-run with --yes to download (multi-GB possible).")


def run_osv(args: argparse.Namespace, output_dir: Path, token: str | None) -> None:
    max_samples = args.max_samples or 10000
    if args.split == "train":
        print(f"[info] OSV-5M default cap -> --max-samples {max_samples}")
    shards = _parse_shards(args.shards, args.split, args.max_shards)

    # Confirmation gate BEFORE any network/download.
    print_osv_plan(args, output_dir)
    if not args.yes:
        print("[gate] aborting before any download. Re-run with --yes to proceed.")
        sys.exit(1)

    tmp = output_dir / "_tmp_osv"
    tmp.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if args.split == "test":
        csv_name = "test.csv"
        print(f"[download] {csv_name} (~{OSV_TEST_CSV_MB}MB, single file, exact and stated)")
        try:
            local_csv = _hub_download(csv_name, tmp, token)
        except Exception as e:
            raise OSVDownloadError(f"Could not download {csv_name}: {e}. "
                                   "The public repo may be rate-limited; retry with an optional HF token if needed.")
        with open(local_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            id_col = _find_col(header, *ID_CANDS)
            path_col = _find_col(header, *PATH_CANDS)
            lat_col = _find_col(header, *LAT_CANDS)
            lon_col = _find_col(header, *LON_CANDS)
            if lat_col is None or lon_col is None:
                raise OSVDownloadError(f"{csv_name} lacks lat/lon columns (got: {header}).")
            extra_cols = [c for c in header if _find_col([c], *EXTRA_CANDS) is not None]
            rows = []
            for r in reader:
                try:
                    lat = float(r[lat_col])
                    lon = float(r[lon_col])
                except (TypeError, ValueError):
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                rec: dict[str, Any] = {"lat": lat, "lon": lon}
                rec["_id"] = str(r[id_col]).strip() if id_col and r.get(id_col) else ""
                rec["_path"] = str(r[path_col]).strip() if path_col and r.get(path_col) else ""
                for c in extra_cols:
                    rec[c] = r.get(c, "")
                rows.append(rec)
        print(f"[meta] {csv_name}: {len(rows)} valid rows")
        if len(rows) > max_samples:
            rows = rng.sample(rows, max_samples)
            print(f"[sample] seeded sample -> {len(rows)} rows (seed {args.seed})")
        wanted: dict[str, dict] = {}
        for r in rows:
            keys = [k for k in (r["_id"], _stem(r["_path"]) if r["_path"] else "") if k]
            if not keys:
                continue
            primary = keys[0]
            if primary not in wanted:
                wanted[primary] = r
                wanted[primary]["_keys"] = keys
        # Download test shards (bounded: at most 5) with HEAD sizes first.
        shard_urls = [(_osv_shard_name("test", s), _hub_url(_osv_shard_name("test", s))) for s in shards]
        print("[shards] planned test shard downloads:")
        sizes = {}
        for name, url in shard_urls:
            sizes[name] = _head_size(url, token)
            print(f"  {name}: {_fmt_bytes(sizes[name])}")
        extracted_total = 0
        missing_all: list[str] = []
        for name, url in shard_urls:
            zpath = _hub_download(name, tmp, token)
            with zipfile.ZipFile(zpath) as zf:
                names = [nm for nm in zf.namelist() if not nm.endswith("/")]
            stems = {_stem(nm) for nm in names}
            todo = [k for keys in (v["_keys"] for v in wanted.values()) for k in keys]
            todo = [t for t in todo if t in stems]
            # Map matched stems back to primary rows for metadata names.
            n, missing = _extract_members(zpath, sorted(set(todo)), images_dir)
            extracted_total += n
            missing_all.extend(missing)
            print(f"[extract] {name}: {len(names)} members, extracted {n} selected")
        # Write metadata only for rows whose image actually landed on disk.
        on_disk = {_stem(p.name) for p in images_dir.iterdir() if p.is_file()}
        out_rows = []
        for primary, r in wanted.items():
            hit = next((k for k in r["_keys"] if k in on_disk), None)
            if hit is None:
                continue
            ext = next((p.suffix for p in images_dir.iterdir() if _stem(p.name) == hit), ".jpg")
            out_rows.append((f"images/{hit}{ext}", r["lat"], r["lon"], primary, "test",
                             *(r.get(c, "") for c in extra_cols)))
            if len(out_rows) >= max_samples:
                break
        _write_metadata(output_dir, out_rows, extra_cols)
        _report_counts(max_samples, len(out_rows), extracted_total, missing_all)
        return

    # ---- split == train ----
    stream_url = _hub_url("train.csv")
    chunk_bytes = args.chunk_mb * 1024 * 1024
    cap_bytes = args.csv_scan_cap_mb * 1024 * 1024
    total_streamed = 0

    if args.allow_full_train_csv:
        print(f"[download] FULL train.csv (~{OSV_TRAIN_CSV_GB}GB, single file, explicit opt-in)")
        local_csv = _hub_download("train.csv", tmp, token)
        with open(local_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            id_col = _find_col(header, *ID_CANDS)
            lat_col = _find_col(header, *LAT_CANDS)
            lon_col = _find_col(header, *LON_CANDS)
            if lat_col is None or lon_col is None:
                raise OSVDownloadError(f"train.csv lacks lat/lon columns (got: {header}).")
            extra_cols = [c for c in header if _find_col([c], *EXTRA_CANDS) is not None]
            pool = []
            for r in reader:
                if id_col is None or not r.get(id_col):
                    continue
                try:
                    lat = float(r[lat_col])
                    lon = float(r[lon_col])
                except (TypeError, ValueError):
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                rec2: dict[str, Any] = {"lat": lat, "lon": lon, "id": str(r[id_col]).strip(), "split": "train"}
                for c in extra_cols:
                    rec2[c] = r.get(c, "")
                pool.append(rec2)
        print(f"[meta] train.csv: {len(pool)} valid rows")
        if len(pool) > max_samples:
            pool = rng.sample(pool, max_samples)
        wanted_ids = [r["id"] for r in pool]
        by_id = {r["id"]: r for r in pool}
    else:
        print(f"[stream] train.csv via Range chunks (cap {args.csv_scan_cap_mb}MB) - full file (~{OSV_TRAIN_CSV_GB}GB) never fetched wholly")
        by_id = {}
        wanted_ids = []
        extra_cols = []
        for shard in shards:
            name = _osv_shard_name("train", shard)
            size = _head_size(_hub_url(name), token)
            print(f"[shard] {name}: {_fmt_bytes(size)}")
            zpath = _hub_download(name, tmp, token)
            with zipfile.ZipFile(zpath) as zf:
                names = [nm for nm in zf.namelist() if not nm.endswith("/")]
            stems = sorted({_stem(nm) for nm in names})
            print(f"[shard] {name}: {len(names)} members, {len(stems)} unique ids")
            new_ids = [s for s in stems if s not in by_id]
            if not new_ids:
                continue
            # Stream-scan for the union of still-unmatched ids (seeded order).
            pending = {s: {} for s in rng.sample(new_ids, len(new_ids))}
            found, scanned, eof = _stream_match_train_csv(
                stream_url, pending, max_samples - len(by_id),
                chunk_bytes, cap_bytes - total_streamed, token)
            total_streamed += scanned
            print(f"[scan] matched {len(found)}/{len(pending)} ids in {_fmt_bytes(scanned)} "
                  f"(cumulative {_fmt_bytes(total_streamed)}, eof={eof})")
            by_id.update(found)
            if found:
                extra_cols = sorted({k for r in found.values() for k in r
                                     if k not in ("lat", "lon", "id", "split", "_src_path")})
            if len(by_id) >= max_samples:
                break
            if total_streamed >= cap_bytes:
                raise OSVDownloadError(
                    f"Scan cap ({args.csv_scan_cap_mb}MB) hit with only {len(by_id)}/{max_samples} rows matched. "
                    "Narrow with --shards, raise --csv-scan-cap-mb, or pass --allow-full-train-csv "
                    "to download the full train.csv (~2.92GB). No images downloaded beyond the listed shards.")
        wanted_ids = list(by_id.keys())[:max_samples]

    # Extract only matched ids from the downloaded shards.
    extracted_total = 0
    missing_all = []
    for shard in shards:
        zpath = tmp / "images" / "train" / f"{shard}.zip"
        alt = tmp / Path(_osv_shard_name("train", shard))
        zp = zpath if zpath.exists() else alt
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp) as zf:
            stems = {_stem(nm) for nm in zf.namelist() if not nm.endswith("/")}
        todo = [i for i in wanted_ids if i in stems]
        n, missing = _extract_members(zp, todo, images_dir)
        extracted_total += n
        print(f"[extract] {zp.name}: extracted {n} selected")
    on_disk = {_stem(p.name) for p in images_dir.iterdir() if p.is_file()}
    out_rows = []
    for i in wanted_ids:
        r = by_id.get(i)
        if r is None or i not in on_disk:
            if i not in on_disk:
                missing_all.append(i)
            continue
        ext = next((p.suffix for p in images_dir.iterdir() if _stem(p.name) == i), ".jpg")
        out_rows.append((f"images/{i}{ext}", r["lat"], r["lon"], i, "train",
                         *(r.get(c, "") for c in extra_cols)))
    _write_metadata(output_dir, out_rows, extra_cols)
    _report_counts(max_samples, len(out_rows), extracted_total, missing_all)


def _write_metadata(output_dir: Path, rows: list[tuple], extra_cols: list[str]) -> Path:
    csv_path = output_dir / "metadata.csv"
    header = ["image_path", "lat", "lon", "id", "split", *extra_cols]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows -> {csv_path}")
    return csv_path


def _report_counts(requested: int, written: int, extracted: int, missing: list[str]) -> None:
    print(f"[counts] requested={requested} extracted_files={extracted} "
          f"metadata_rows={written} missing={len(missing)}")
    if missing[:10]:
        print(f"  missing ids (first {min(10, len(missing))}): {', '.join(missing[:10])}")
    if written < requested:
        print(f"[honesty] Only {written} rows extracted (requested {requested}). "
              "The subset is smaller than asked - NOT a full '10k'. See missing ids above.")
    else:
        print(f"[done] {written} rows ready.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    print("[info] KPlaceNet safe downloader (no snapshot_download, no full-repo fetch)")
    print(f"  dataset={args.dataset} -> {output_dir}")

    if args.dataset == "im2gps3k":
        print("[note] IM2GPS3k is eval-only, never train (plan Sec 2). 3k benchmark.")
        print("  Manual step: download from the authors and place images + metadata.csv")
        print(f"  under {output_dir} with header image_path,lat,lon. No automated fetch.")
        sys.exit(2)

    if args.dataset == "flickr_geo_tiny":
        if args.dry_run:
            n = args.max_samples if args.max_samples is not None else 50
            make_dummy_csv(output_dir, n=n, seed=args.seed)
            print("[done] dry-run complete. Smoke train with:")
            print(f"  python -m src.train --csv {output_dir / 'metadata.csv'} --epochs 2 --batch-size 32 --amp")
            return
        print("[error] No real Flickr-Geo Hub repo is configured (placeholder id).")
        print("  Use --dry-run for the dummy smoke set, or place real images manually.")
        sys.exit(2)

    # ---- osv5m: safe path only ----
    if args.max_samples is None:
        args.max_samples = 10000
    if args.dry_run or args.plan_only:
        try:
            print_osv_plan(args, output_dir)
        except OSVDownloadError as e:
            print(f"[error] {e}")
            sys.exit(2)
        print("[done] plan only - no network, no files written.")
        return
    token = _token(args)
    if token is None:
        print("[note] No HF token (--hf-token or HF_TOKEN). The public repo may still be rate-limited.")
    try:
        run_osv(args, output_dir, token)
    except OSVDownloadError as e:
        print(f"[error] {e}")
        sys.exit(2)
    print("[done] download complete.")


if __name__ == "__main__":
    main()
