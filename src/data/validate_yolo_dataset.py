from pathlib import Path
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "yolo_road_damage"

CLASS_IDS = {0, 1, 2, 3}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def validate_split(split):
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = image_files(image_dir)

    missing_labels = []
    invalid_labels = []
    bad_images = []

    for image in images:

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing_labels.append(image.name)
            continue

        try:
            with Image.open(image) as im:
                im.verify()
        except Exception:
            bad_images.append(image.name)
            continue

        for line_number, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(),
            start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                invalid_labels.append(
                    f"{label.name}: line {line_number}: wrong field count"
                )
                continue

            try:
                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]
            except ValueError:
                invalid_labels.append(
                    f"{label.name}: line {line_number}: non-numeric value"
                )
                continue

            x, y, width, height = values

            if class_id not in CLASS_IDS:
                invalid_labels.append(
                    f"{label.name}: line {line_number}: invalid class {class_id}"
                )

            if not all(0 <= value <= 1 for value in values):
                invalid_labels.append(
                    f"{label.name}: line {line_number}: coordinates outside 0..1"
                )

            if width <= 0 or height <= 0:
                invalid_labels.append(
                    f"{label.name}: line {line_number}: invalid box size"
                )

    return {
        "images": images,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "bad_images": bad_images,
    }


def main():

    print("=" * 65)
    print("NagarSetu - YOLO DATASET VALIDATION")
    print("=" * 65)

    if not DATASET.exists():
        print(f"ERROR: dataset not found:\n{DATASET}")
        return

    train = validate_split("train")
    val = validate_split("val")

    print("\nTRAIN")
    print(f"Images              : {len(train['images']):,}")
    print(f"Missing labels      : {len(train['missing_labels']):,}")
    print(f"Invalid label lines : {len(train['invalid_labels']):,}")
    print(f"Bad images          : {len(train['bad_images']):,}")

    print("\nVALIDATION")
    print(f"Images              : {len(val['images']):,}")
    print(f"Missing labels      : {len(val['missing_labels']):,}")
    print(f"Invalid label lines : {len(val['invalid_labels']):,}")
    print(f"Bad images          : {len(val['bad_images']):,}")

    train_names = {p.name for p in train["images"]}
    val_names = {p.name for p in val["images"]}

    overlap = train_names & val_names

    print("\nSPLIT LEAKAGE CHECK")
    print(f"Train/validation overlap: {len(overlap)}")

    errors = (
        len(train["missing_labels"])
        + len(train["invalid_labels"])
        + len(train["bad_images"])
        + len(val["missing_labels"])
        + len(val["invalid_labels"])
        + len(val["bad_images"])
        + len(overlap)
    )

    print("\n" + "=" * 65)

    if errors == 0:
        print("✅ YOLO DATASET VALIDATION PASSED")
        print("The dataset is structurally ready for training.")
    else:
        print(f"❌ VALIDATION FAILED — {errors} problem(s) found.")
        print("Fix the dataset before training.")

    print("=" * 65)


if __name__ == "__main__":
    main()
