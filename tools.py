#!/usr/bin/env python3
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, List

_CHUNK = 8 * 1024 * 1024  # 8 MiB


def validate_parts(parts: Iterable[Path]) -> bool:
    return all(Path(p).exists() for p in parts)


def _sorted_parts(parts: Iterable[Path]) -> List[Path]:
    def num(p: Path):
        m = re.search(r"_part(\d+)\.zip$", p.name)
        return int(m.group(1)) if m else 10**9
    return sorted(map(Path, parts), key=num)


def combine_and_extract(parts: Iterable[Path], extract_to: Path, combined_zip_path: Path) -> None:
    parts = _sorted_parts(parts)
    extract_to = Path(extract_to)
    combined_zip_path = Path(combined_zip_path)
    extract_to.mkdir(parents=True, exist_ok=True)
    combined_zip_path.parent.mkdir(parents=True, exist_ok=True)

    # combine
    with open(combined_zip_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    buf = f.read(_CHUNK)
                    if not buf:
                        break
                    out.write(buf)

    # extract
    with zipfile.ZipFile(combined_zip_path, "r") as z:
        z.extractall(extract_to)

    # cleanup
    try:
        os.remove(combined_zip_path)
    except OSError:
        pass
