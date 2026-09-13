from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "road_damage_annotations.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

ANNOTATION_OUTPUT = (
    OUTPUT_DIR
    / "road_damage_v1_annotations.csv"
)

IMAGE_OUTPUT = (
    OUTPUT_DIR
    / "road_damage_v1_images.csv"
)


OFFICIAL_CLASSES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}


def main():

    print("=" * 60)
    print("NagarSetu - Building Road Damage V1 Dataset")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print("ERROR: Input annotation file not found.")
        print(INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"Original annotation rows : {len(df):,}")
    print(f"Original unique images   : {df['image_file'].nunique():,}")

    # Keep only the four official CRDDC classes
    df = df[df["damage_class"].isin(OFFICIAL_CLASSES)].copy()

    # Human-readable label
    df["NagarSetu_class"] = df["damage_class"].map(OFFICIAL_CLASSES)

    # Image path relative to the dataset
    df["image_path"] = (
        "data/raw/road_damage/India/India/"
        "train/images/"
        + df["image_file"].astype(str)
    )

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Save annotation-level dataset
    df.to_csv(
        ANNOTATION_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------
    # Create image-level summary
    # ------------------------------------------------------

    image_df = (
        df.groupby("image_file")
        .agg(
            annotation_count=("image_file", "size"),
            classes=("NagarSetu_class", lambda x: "|".join(sorted(set(x)))),
            primary_class=(
                "NagarSetu_class",
                lambda x: x.value_counts().index[0]
            ),
            max_damage_area_ratio=(
                "damage_area_ratio",
                "max"
            ),
        )
        .reset_index()
    )

    image_df["image_path"] = (
        "data/raw/road_damage/India/India/"
        "train/images/"
        + image_df["image_file"].astype(str)
    )

    image_df["dataset"] = "RDD2022"
    image_df["country"] = "India"
    image_df["split"] = "train"

    image_df.to_csv(
        IMAGE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("CLEAN DATASET SUMMARY")
    print("=" * 60)

    print(f"Annotations : {len(df):,}")
    print(f"Images      : {df['image_file'].nunique():,}")

    print()
    print("Class distribution:")
    print(
        df["NagarSetu_class"]
        .value_counts()
        .to_string()
    )

    print()
    print("Saved annotation dataset:")
    print(ANNOTATION_OUTPUT)

    print()
    print("Saved image-level dataset:")
    print(IMAGE_OUTPUT)


if __name__ == "__main__":
    main()
