#!/usr/bin/env python3
"""Build a flat-ground-only A3 PKL dataset.

The source dataset is expected to be organized as:
    <source>/<motion_dir>/data_pkl/*.pkl

The output dataset is organized as:
    <output>/data_pkl/*.pkl

The script is intentionally conservative about text matching. It rejects only
explicit phrases/tokens that imply non-flat terrain, external objects/props, or
sit/box/parkour style interactions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SOURCE = Path("a3_data/base_model_data_original")
DEFAULT_OUTPUT = Path("a3_data/base_model_data_flat_ground")

REJECT_RULES: dict[str, tuple[str, ...]] = {
    "terrain_stair_slope": (
        "shangpo",
        "xiapo",
        "slope",
        "shanglouti",
        "xialouti",
        "xialou",
        "shanglou",
        "上坡",
        "下坡",
        "楼梯",
    ),
    "box_parkour": (
        "ruxiang",
        "chuxiang",
        "paxiangzi",
        "tiaoxiangzi",
        "fanxiangzi",
        "xiangzi",
        "shangxiang",
        "paoku",
        "parkour",
        "入箱",
        "出箱",
        "箱子",
        "爬箱",
        "跳箱",
        "跑酷",
    ),
    "hard_motion": (
        "bmd_0120_a3_filtered_20260120_193312_2_ceti_skeleton0",
        "bmd_0124",
        "bmd_0122_a3_filtered_20260122_133745_poes5tongzibaifo_wushu",
        "bmd_0122_a3_mirror_filtered_20260226_143727_poes5tongzibaifo_wushu",
        "bmd_0122_a3_filtered_20260226_152730_poes5tongzibaifo_wushu",
        "bmd_0311_a3_filtered_20260312_102626_jump_004_skeleton0",
        "bmd_0311_a3_mirror_filtered_20260312_113047_jump_004_skeleton0",
        "bmd_1213_a3_filtered_20260128_162239_quanji",
        "bmd_1213_a3_mirror_filtered_20260128_191629_quanji",
        "feiti",
        "gongfu",
        "gongfutaolu",
        "kongfan",
        "shibatongren",
        "wushudongzuo",
    ),
    "sit_or_sit_to_stand": (
        "zuoxia",
        "sit_down",
        "sit_to_stand",
        "sitdown",
        "sitting",
        "sit_high",
        "sit_walk",
        "8ziyou_man_long",
        "youcezhuan_kuai_long",
        "坐下",
    ),
}

MANIFEST_FIELDS = (
    "source_rel_path",
    "source_dir",
    "source_file",
    "sha256",
    "status",
    "reject_categories",
    "reason",
    "basename_duplicate_count",
    "hash_duplicate_count",
    "duplicate_action",
    "representative_rel_path",
    "output_filename",
    "output_rel_path",
)

SUMMARY_FIELDS = (
    "source_dir",
    "total",
    "keep",
    "reject",
    "uncertain",
    "copied_files",
    "deduped_same_content",
)


@dataclass
class Record:
    source_path: Path
    source_root: Path
    rel_path: str
    source_dir: str
    source_file: str
    sha256: str
    status: str
    reject_categories: list[str]
    reason: str
    basename_duplicate_count: int = 1
    hash_duplicate_count: int = 1
    duplicate_action: str = ""
    representative_rel_path: str = ""
    output_filename: str = ""
    output_rel_path: str = ""

    def as_manifest_row(self) -> dict[str, str]:
        return {
            "source_rel_path": self.rel_path,
            "source_dir": self.source_dir,
            "source_file": self.source_file,
            "sha256": self.sha256,
            "status": self.status,
            "reject_categories": ";".join(self.reject_categories),
            "reason": self.reason,
            "basename_duplicate_count": str(self.basename_duplicate_count),
            "hash_duplicate_count": str(self.hash_duplicate_count),
            "duplicate_action": self.duplicate_action,
            "representative_rel_path": self.representative_rel_path,
            "output_filename": self.output_filename,
            "output_rel_path": self.output_rel_path,
        }


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", text.lower())


def classify_file(source_dir: str, source_file: str) -> tuple[str, list[str], str]:
    text = normalize_for_match(f"{source_dir}/{source_file}")
    categories = [
        category
        for category, needles in REJECT_RULES.items()
        if any(needle.lower() in text for needle in needles)
    ]
    if categories:
        return "reject", categories, "matched strong reject keyword(s)"
    return "keep", [], "no reject keyword matched"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_output_name(
    desired_name: str,
    used_names: set[str],
    source_dir: str,
    source_file: str,
    file_hash: str,
) -> str:
    candidate = desired_name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    stem = Path(source_file).stem
    suffix = Path(source_file).suffix
    candidate = f"{source_dir}__{stem}__{file_hash[:12]}{suffix}"
    counter = 2
    while candidate in used_names:
        candidate = f"{source_dir}__{stem}__{file_hash[:12]}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def collect_records(source_root: Path) -> list[Record]:
    files = sorted(source_root.glob("*/data_pkl/*.pkl"), key=lambda p: str(p.relative_to(source_root)))
    records: list[Record] = []
    for path in files:
        rel_path = path.relative_to(source_root).as_posix()
        source_dir = path.parts[-3]
        source_file = path.name
        status, categories, reason = classify_file(source_dir, source_file)
        records.append(
            Record(
                source_path=path,
                source_root=source_root,
                rel_path=rel_path,
                source_dir=source_dir,
                source_file=source_file,
                sha256=sha256_file(path),
                status=status,
                reject_categories=categories,
                reason=reason,
            )
        )
    return records


def assign_outputs(records: list[Record]) -> None:
    basename_counts = Counter(record.source_file for record in records)
    keep_records = [record for record in records if record.status == "keep"]
    keep_by_basename: dict[str, list[Record]] = defaultdict(list)
    for record in keep_records:
        record.basename_duplicate_count = basename_counts[record.source_file]
        keep_by_basename[record.source_file].append(record)

    used_names: set[str] = set()
    for basename, same_name_records in sorted(keep_by_basename.items()):
        by_hash: dict[str, list[Record]] = defaultdict(list)
        for record in same_name_records:
            by_hash[record.sha256].append(record)

        basename_is_unique = len(same_name_records) == 1
        basename_is_same_content = len(by_hash) == 1

        for file_hash, same_hash_records in sorted(by_hash.items()):
            same_hash_records.sort(key=lambda record: record.rel_path)
            representative = same_hash_records[0]
            for record in same_hash_records:
                record.hash_duplicate_count = len(same_hash_records)
                record.representative_rel_path = representative.rel_path

            if basename_is_unique or basename_is_same_content:
                desired = basename
            else:
                desired = f"{representative.source_dir}__{basename}"

            output_name = unique_output_name(
                desired,
                used_names,
                representative.source_dir,
                representative.source_file,
                representative.sha256,
            )
            representative.output_filename = output_name
            representative.output_rel_path = f"data_pkl/{output_name}"
            representative.duplicate_action = "copy"

            for duplicate in same_hash_records[1:]:
                duplicate.output_filename = output_name
                duplicate.output_rel_path = f"data_pkl/{output_name}"
                duplicate.duplicate_action = "dedup_same_content"


def write_manifest(records: list[Record], output_root: Path) -> None:
    with (output_root / "filter_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.as_manifest_row() for record in records)


def write_summary(records: list[Record], output_root: Path) -> None:
    by_dir: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_dir[record.source_dir].append(record)

    with (output_root / "filter_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for source_dir in sorted(by_dir):
            dir_records = by_dir[source_dir]
            status_counts = Counter(record.status for record in dir_records)
            writer.writerow(
                {
                    "source_dir": source_dir,
                    "total": len(dir_records),
                    "keep": status_counts.get("keep", 0),
                    "reject": status_counts.get("reject", 0),
                    "uncertain": status_counts.get("uncertain", 0),
                    "copied_files": sum(
                        1 for record in dir_records if record.duplicate_action == "copy"
                    ),
                    "deduped_same_content": sum(
                        1
                        for record in dir_records
                        if record.duplicate_action == "dedup_same_content"
                    ),
                }
            )


def source_dirs_from_meta(row: dict[str, str]) -> list[str]:
    return [item for item in row.get("匹配目录", "").split(";") if item]


def write_filtered_meta(records: list[Record], source_root: Path, output_root: Path) -> None:
    source_meta = source_root / "meta.csv"
    if not source_meta.exists():
        return

    rows = list(csv.DictReader(source_meta.open(encoding="utf-8", newline="")))
    if not rows:
        return

    records_by_dir: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        records_by_dir[record.source_dir].append(record)

    base_fields = list(rows[0].keys())
    extra_fields = [
        "筛选前数量",
        "筛选后数量",
        "筛选剔除数量",
        "筛选不确定数量",
        "筛选输出文件",
    ]
    fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]

    output_rows: list[dict[str, str]] = []
    for row in rows:
        source_dirs = source_dirs_from_meta(row)
        related = [record for source_dir in source_dirs for record in records_by_dir.get(source_dir, [])]
        output_files = sorted(
            {
                record.output_rel_path
                for record in related
                if record.status == "keep" and record.output_rel_path
            }
        )
        reject_count = sum(1 for record in related if record.status == "reject")
        uncertain_count = sum(1 for record in related if record.status == "uncertain")
        before_count = row.get("实际数量") or row.get("数量") or str(len(related))

        new_row = dict(row)
        new_row["筛选前数量"] = before_count
        new_row["筛选后数量"] = str(len(output_files))
        new_row["筛选剔除数量"] = str(reject_count)
        new_row["筛选不确定数量"] = str(uncertain_count)
        new_row["筛选输出文件"] = ";".join(output_files)
        new_row["数量"] = str(len(output_files))
        new_row["实际数量"] = str(len(output_files))
        output_rows.append(new_row)

    with (output_root / "meta.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)


def copy_outputs(records: list[Record], output_root: Path) -> None:
    data_dir = output_root / "data_pkl"
    data_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.status != "keep" or record.duplicate_action != "copy":
            continue
        shutil.copy2(record.source_path, data_dir / record.output_filename)


def validate(records: list[Record], source_root: Path, output_root: Path, apply: bool) -> dict[str, int]:
    total = len(records)
    status_counts = Counter(record.status for record in records)
    if status_counts["keep"] + status_counts["reject"] + status_counts["uncertain"] != total:
        raise RuntimeError("manifest statuses do not add up to total PKL count")

    output_names = [
        record.output_filename
        for record in records
        if record.status == "keep" and record.duplicate_action == "copy"
    ]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError("output filenames are not unique")

    if apply:
        output_files = sorted((output_root / "data_pkl").glob("*.pkl"))
        expected = len(output_names)
        if len(output_files) != expected:
            raise RuntimeError(f"output file count mismatch: expected {expected}, got {len(output_files)}")

        copied_names = {path.name for path in output_files}
        missing = [name for name in output_names if name not in copied_names]
        if missing:
            raise RuntimeError(f"missing copied output files: {missing[:10]}")

        stray = [
            path
            for path in output_root.rglob("*")
            if path.is_file()
            and path.suffix == ".pkl"
            and path.parent != output_root / "data_pkl"
        ]
        if stray:
            raise RuntimeError(f"PKL files found outside data_pkl: {stray[:10]}")

    source_count_after = sum(1 for _ in source_root.glob("*/data_pkl/*.pkl"))
    if source_count_after != total:
        raise RuntimeError(
            f"source PKL count changed during filtering: before {total}, after {source_count_after}"
        )

    return {
        "total": total,
        "keep": status_counts["keep"],
        "reject": status_counts["reject"],
        "uncertain": status_counts["uncertain"],
        "copied": sum(1 for record in records if record.duplicate_action == "copy"),
        "deduped_same_content": sum(
            1 for record in records if record.duplicate_action == "dedup_same_content"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true", help="copy kept PKLs into output/data_pkl")
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove the output directory before writing. Required if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if not source_root.exists():
        raise SystemExit(f"source directory does not exist: {source_root}")
    if output_root.exists():
        if not args.force:
            raise SystemExit(f"output exists; rerun with --force to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    records = collect_records(source_root)
    assign_outputs(records)
    write_manifest(records, output_root)
    write_summary(records, output_root)
    write_filtered_meta(records, source_root, output_root)
    if args.apply:
        copy_outputs(records, output_root)

    stats = validate(records, source_root, output_root, args.apply)
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"output={output_root}")


if __name__ == "__main__":
    main()
