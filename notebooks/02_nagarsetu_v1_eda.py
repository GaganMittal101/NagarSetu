from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "road_damage_v1_annotations.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "eda_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 70)
    print("NagarSetu - V1 ROAD DAMAGE EDA")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nAnnotations : {len(df):,}")
    print(f"Images      : {df['image_file'].nunique():,}")
    print(f"Classes     : {df['NagarSetu_class'].nunique()}")

    # ----------------------------
    # Class distribution
    # ----------------------------

    class_counts = (
        df["NagarSetu_class"]
        .value_counts()
    )

    class_pct = (
        df["NagarSetu_class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    distribution = pd.DataFrame({
        "count": class_counts,
        "percentage": class_pct
    })

    print("\nCLASS DISTRIBUTION")
    print(distribution)

    distribution.to_csv(
        OUTPUT_DIR / "class_distribution.csv"
    )

    # ----------------------------
    # Images per class
    # ----------------------------

    images_per_class = (
        df.groupby("NagarSetu_class")["image_file"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\nUNIQUE IMAGES PER CLASS")
    print(images_per_class)

    images_per_class.to_csv(
        OUTPUT_DIR / "images_per_class.csv"
    )

    # ----------------------------
    # Multiple damage analysis
    # ----------------------------

    annotations_per_image = (
        df.groupby("image_file")
        .size()
    )

    print("\nANNOTATIONS PER IMAGE")
    print(
        annotations_per_image.describe()
    )

    print(
        "\nImages with multiple damages:",
        int((annotations_per_image > 1).sum())
    )

    # ----------------------------
    # Bounding boxes
    # ----------------------------

    bbox_stats = (
        df.groupby("NagarSetu_class")[
            [
                "box_width",
                "box_height",
                "box_area",
                "damage_area_ratio",
            ]
        ]
        .agg(["mean", "median"])
        .round(4)
    )

    print("\nBOUNDING BOX STATISTICS")
    print(bbox_stats)

    bbox_stats.to_csv(
        OUTPUT_DIR / "bbox_statistics.csv"
    )

    # ----------------------------
    # Class plot
    # ----------------------------

    plt.figure(figsize=(9, 5))

    class_counts.plot(
        kind="bar"
    )

    plt.title(
        "NagarSetu V1 - Road Damage Distribution"
    )

    plt.xlabel("Damage Class")
    plt.ylabel("Annotations")

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "NagarSetu_v1_class_distribution.png",
        dpi=160
    )

    plt.close()

    # ----------------------------
    # Damage-area plot
    # ----------------------------

    plt.figure(figsize=(9, 5))

    df.boxplot(
        column="damage_area_ratio",
        by="NagarSetu_class",
        rot=20,
    )

    plt.title(
        "Damage Area Ratio by NagarSetu Class"
    )

    plt.suptitle("")

    plt.xlabel("Damage Class")
    plt.ylabel("Damage / Image Area")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "damage_area_by_class.png",
        dpi=160
    )

    plt.close()

    # ----------------------------
    # Summary
    # ----------------------------

    summary = {
        "annotations": len(df),
        "images": df["image_file"].nunique(),
        "classes": df["NagarSetu_class"].nunique(),
        "multi_damage_images": int(
            (annotations_per_image > 1).sum()
        ),
        "max_annotations_per_image": int(
            annotations_per_image.max()
        ),
        "median_damage_area_ratio": float(
            df["damage_area_ratio"].median()
        ),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("V1 EDA COMPLETE")
    print("=" * 70)

    print(
        f"\nOutputs saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

