"""Checks that both datasets are in place and every file opens.

Run from the project root:
    python scripts/check_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import list_cases, load_case, load_image, load_planes_table


def check_focus():
    print("FOCUS")
    for split in ("train", "val", "test"):
        names = list_cases(split)
        errors = 0
        for name in names:
            try:
                load_case(split, name)  # image + ellipse labels + masks
            except Exception as e:
                errors += 1
                print(f"  {split}/{name}: {e}")
        print(f"  {split:5s}: {len(names)} images, {errors} errors")


def check_planes():
    print("FETAL_PLANES_DB")
    table = load_planes_table()
    errors = 0
    for path in table["image_path"]:
        try:
            load_image(path)
        except Exception as e:
            errors += 1
            print(f"  {e}")
    print(f"  {len(table)} images in the table, {errors} errors")


if __name__ == "__main__":
    check_focus()
    check_planes()
