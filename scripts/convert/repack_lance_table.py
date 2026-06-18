"""Repack one Lance table with a stricter file-size target."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import lance
import pyarrow as pa

from vlm2emb.data.datasets.const import LANCE_MAX_BYTES_PER_FILE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source Lance table path.")
    parser.add_argument("--target", type=Path, required=True, help="Target Lance table path.")
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=LANCE_MAX_BYTES_PER_FILE,
        help="Target maximum Lance data file size.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Scanner/write batch size.")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        required=True,
        help="Directory that receives replaced source/target tables.",
    )
    parser.add_argument(
        "--move-source-to-backup",
        action="store_true",
        help="Move source to backup after writing when source and target differ.",
    )
    return parser.parse_args()


def _iter_batches(dataset: lance.LanceDataset, *, batch_size: int) -> pa.RecordBatchReader:
    schema = dataset.schema

    def batches():
        for batch in dataset.scanner(batch_size=batch_size).to_batches():
            yield batch

    return pa.RecordBatchReader.from_batches(schema, batches())


def _backup_path(path: Path, backup_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = backup_dir / f"{path.name}.{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = backup_dir / f"{path.name}.{timestamp}.{suffix}"
        suffix += 1
    return candidate


def main() -> None:
    args = _parse_args()
    source = args.source
    target = args.target
    backup_dir = args.backup_dir
    tmp_target = target.with_name(f".{target.name}.repack-tmp-{int(time.time())}")

    if not source.exists():
        raise FileNotFoundError(f"Missing source Lance table: {source}")
    if tmp_target.exists():
        raise FileExistsError(f"Temporary target already exists: {tmp_target}")

    started_at = time.monotonic()
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    dataset = lance.dataset(str(source))
    row_count = dataset.count_rows()
    print(
        json.dumps(
            {
                "event": "repack_lance_start",
                "source": str(source),
                "target": str(target),
                "tmp_target": str(tmp_target),
                "rows": row_count,
                "batch_size": args.batch_size,
                "max_bytes_per_file": args.max_bytes_per_file,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    lance.write_dataset(
        _iter_batches(dataset, batch_size=args.batch_size),
        str(tmp_target),
        schema=dataset.schema,
        mode="create",
        max_bytes_per_file=args.max_bytes_per_file,
    )

    replaced: list[dict[str, str]] = []
    if target.exists():
        backup = _backup_path(target, backup_dir)
        shutil.move(str(target), str(backup))
        replaced.append({"path": str(target), "backup": str(backup)})
    if args.move_source_to_backup and source != target and source.exists():
        backup = _backup_path(source, backup_dir)
        shutil.move(str(source), str(backup))
        replaced.append({"path": str(source), "backup": str(backup)})

    shutil.move(str(tmp_target), str(target))
    output_dataset = lance.dataset(str(target))
    print(
        json.dumps(
            {
                "event": "repack_lance_done",
                "source": str(source),
                "target": str(target),
                "rows": output_dataset.count_rows(),
                "replaced": replaced,
                "elapsed_sec": round(time.monotonic() - started_at, 3),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
