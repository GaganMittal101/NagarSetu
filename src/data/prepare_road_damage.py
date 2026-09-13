from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


# ============================================================
# NagarSetu
# RDD2022 India -> structured annotation CSV
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INDIA_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "road_damage"
    / "India"
    / "India"
)

TRAIN_IMAGE_DIR = INDIA_ROOT / "train" / "images"
XML_DIR = INDIA_ROOT / "train" / "annotations" / "xmls"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "road_damage_annotations.csv"


def safe_int(node):
    """Convert XML text to integer safely."""
    if node is None or node.text is None:
        return None

    try:
        return int(node.text)
    except ValueError:
        return None


def parse_xml(xml_path: Path) -> list[dict]:
    """Read one Pascal-VOC XML file."""

    rows = []

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        print(f"Skipping invalid XML: {xml_path.name} -> {exc}")
        return rows

    filename_node = root.find("filename")

    if filename_node is None or not filename_node.text:
        return rows

    image_name = filename_node.text.strip()

    # Image dimensions
    size_node = root.find("size")

    image_width = None
    image_height = None

    if size_node is not None:
        image_width = safe_int(size_node.find("width"))
        image_height = safe_int(size_node.find("height"))

    # Each <object> is one damage annotation
    for obj in root.findall("object"):

        class_node = obj.find("name")

        if class_node is None or not class_node.text:
            continue

        damage_class = class_node.text.strip()

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = safe_int(bbox.find("xmin"))
        ymin = safe_int(bbox.find("ymin"))
        xmax = safe_int(bbox.find("xmax"))
        ymax = safe_int(bbox.find("ymax"))

        if None in [xmin, ymin, xmax, ymax]:
            continue

        box_width = xmax - xmin
        box_height = ymax - ymin

        if box_width <= 0 or box_height <= 0:
            continue

        box_area = box_width * box_height

        image_area = None
        damage_area_ratio = None

        if image_width and image_height:
            image_area = image_width * image_height
            damage_area_ratio = box_area / image_area

        rows.append(
            {
                "image_file": image_name,
                "annotation_file": xml_path.name,
                "damage_class": damage_class,

                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,

                "image_width": image_width,
                "image_height": image_height,

                "box_width": box_width,
                "box_height": box_height,
                "box_area": box_area,

                "damage_area_ratio": damage_area_ratio,

                "country": "India",
                "dataset": "RDD2022",
                "split": "train",
            }
        )

    return rows


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not XML_DIR.exists():
        print("ERROR: XML annotation directory not found.")
        print(XML_DIR)
        return

    if not TRAIN_IMAGE_DIR.exists():
        print("ERROR: training image directory not found.")
        print(TRAIN_IMAGE_DIR)
        return

    xml_files = sorted(XML_DIR.glob("*.xml"))

    print("=" * 60)
    print("NagarSetu - RDD2022 India Data Preparation")
    print("=" * 60)

    print(f"XML annotation files : {len(xml_files):,}")

    if not xml_files:
        print("No XML files found.")
        return

    all_rows = []

    for index, xml_file in enumerate(xml_files, start=1):

        rows = parse_xml(xml_file)
        all_rows.extend(rows)

        if index % 500 == 0:
            print(f"Processed {index:,} XML files...")

    if not all_rows:
        print("No valid annotations were found.")
        return

    df = pd.DataFrame(all_rows)

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Summary
    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Annotation rows : {len(df):,}")
    print(f"Unique images   : {df['image_file'].nunique():,}")
    print(f"Damage classes  : {df['damage_class'].nunique()}")

    print()
    print("Damage class distribution:")
    print(df["damage_class"].value_counts())

    print()
    print("Bounding-box statistics:")
    print(df[[
        "box_width",
        "box_height",
        "box_area",
        "damage_area_ratio"
    ]].describe())

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
