#!/usr/bin/env python3
from invoke import task
from pathlib import Path
from tools import combine_and_extract, validate_parts
import gui as graphical_interface  # keep as-is if you have it


BASE = Path(__file__).parent.resolve()


@task
def extract(c):
    # -------- Dataset (2 parts) --------
    ds_dir = BASE / "training" / "dataset"
    ds_extract = ds_dir / "ships_in_satellite_imagery"
    ds_parts = [ds_dir / f"ships_in_satellite_imagery_part{i}.zip" for i in (1, 2)]
    if validate_parts(ds_parts):
        combine_and_extract(ds_parts, ds_extract, ds_dir / "_combined_dataset_temp.zip")
        print(f"OK: dataset → {ds_extract}")
    else:
        print("Dataset parts missing; skipped.")

    # -------- Models (5 parts) --------
    pre_dir = BASE / "training" / "pretrained"
    model_parts = [pre_dir / f"models_bundle_part{i}.zip" for i in range(1, 6)]
    if validate_parts(model_parts):
        combine_and_extract(model_parts, pre_dir, pre_dir / "_combined_models_temp.zip")
        print(f"OK: models → {pre_dir}")
    else:
        print("Model parts missing; skipped.")


@task
def gui(c):
    graphical_interface.main()
