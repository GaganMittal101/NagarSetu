from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS = ROOT / "data" / "processed" / "road_damage_v1_annotations.csv"

IMAGE_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "road_damage"
    / "India"
    / "India"
    / "train"
    / "images"
)

YOLO_ROOT = ROOT / "data" / "yolo_road_damage"

CLASS_MAP = {
    "Longitudinal Crack": 0,
    "Transverse Crack": 1,
    "Alligator Crack": 2,
    "Pothole": 3,
}


def convert_bbox(row):
    """Convert VOC xyxy coordinates to YOLO normalized format."""

    width = row["image_width"]
    height = row["image_height"]

    x_center = ((row["xmin"] + row["xmax"]) / 2) / width
    y_center = ((row["ymin"] + row["ymax"]) / 2) / height

    box_width = (row["xmax"] - row["xmin"]) / width
    box_height = (row["ymax"] - row["ymin"]) / height

    return x_center, y_center, box_width, box_height


def main():

    print("=" * 65)
    print("NagarSetu - YOLO DATASET PREPARATION")
    print("=" * 65)

    df = pd.read_csv(ANNOTATIONS)

    # -------------------------------------------------------
    # Image-level dataframe
    # -------------------------------------------------------

    image_df = (
        df.groupby("image_file")
        .agg(
            primary_class=(
                "NagarSetu_class",
                lambda x: x.value_counts().index[0]
            )
        )
        .reset_index()
    )

    print(f"Images available: {len(image_df):,}")

    print("\nImage-level class distribution:")
    print(image_df["primary_class"].value_counts())

    # -------------------------------------------------------
    # Train / validation split
    # -------------------------------------------------------

    train_images, val_images = train_test_split(
        image_df,
        test_size=0.20,
        random_state=42,
        stratify=image_df["primary_class"],
    )

    train_set = set(train_images["image_file"])
    val_set = set(val_images["image_file"])

    print()
    print(f"Training images  : {len(train_set):,}")
    print(f"Validation images: {len(val_set):,}")

    # -------------------------------------------------------
    # Create directories
    # -------------------------------------------------------

    train_img_dir = YOLO_ROOT / "images" / "train"
    val_img_dir = YOLO_ROOT / "images" / "val"

    train_lbl_dir = YOLO_ROOT / "labels" / "train"
    val_lbl_dir = YOLO_ROOT / "labels" / "val"

    for directory in [
        train_img_dir,
        val_img_dir,
        train_lbl_dir,
        val_lbl_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # -------------------------------------------------------
    # Process annotations
    # -------------------------------------------------------

    for image_name in image_df["image_file"]:

        source_image = IMAGE_ROOT / image_name

        if not source_image.exists():
            print(f"WARNING: missing image: {image_name}")
            continue

        if image_name in train_set:
            image_out = train_img_dir / image_name
            label_out = train_lbl_dir / (
                Path(image_name).stem + ".txt"
            )
        else:
            image_out = val_img_dir / image_name
            label_out = val_lbl_dir / (
                Path(image_name).stem + ".txt"
            )

        shutil.copy2(
            source_image,
            image_out
        )

        image_annotations = df[
            df["image_file"] == image_name
        ]

        with label_out.open(
            "w",
            encoding="utf-8"
        ) as f:

            for _, row in image_annotations.iterrows():

                class_id = CLASS_MAP[
                    row["NagarSetu_class"]
                ]

                x_center, y_center, box_width, box_height = (
                    convert_bbox(row)
                )

                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )

    # -------------------------------------------------------
    # YOLO dataset configuration
    # -------------------------------------------------------

    yaml_content = """path: data/yolo_road_damage

train: images/train
val: images/val

names:
  0: Longitudinal Crack
  1: Transverse Crack
  2: Alligator Crack
  3: Pothole
"""

    (YOLO_ROOT / "data.yaml").write_text(
        yaml_content,
        encoding="utf-8"
    )

    print()
    print("=" * 65)
    print("YOLO DATASET READY")
    print("=" * 65)

    print(f"Dataset: {YOLO_ROOT}")

    print()
    print("Structure:")
    print("images/train")
    print("images/val")
    print("labels/train")
    print("labels/val")
    print("data.yaml")


if __name__ == "__main__":
    main()
