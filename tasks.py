#!/usr/bin/env python3
from invoke import task
from pathlib import Path
import shutil
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

    # -------- Copy key model weights to spark/app --------
    spark_app = BASE / "spark" / "app"
    spark_app.mkdir(parents=True, exist_ok=True)

    convnext_src = BASE / "training" / "pretrained" / "ConvNext" / "convnext_ships.pt"
    realesrgan_src = BASE / "training" / "pretrained" / "RealESRGAN" /"RealESRGAN_x4plus.pth"

    for src in (convnext_src, realesrgan_src):
        if src.exists():
            shutil.copy2(src, spark_app)
            print(f"Copied {src.name} → {spark_app}")
        else:
            print(f"Missing {src.name}; not copied.")


@task
def gui(c):
    graphical_interface.main()
