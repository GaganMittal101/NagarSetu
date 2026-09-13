from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "models"
    / "runs"
    / "NagarSetu_road_damage_baseline"
    / "weights"
    / "best.pt"
)

SOURCE = (
    ROOT
    / "data"
    / "yolo_road_damage"
    / "images"
    / "val"
)

OUTPUT = ROOT / "models" / "sample_predictions"


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL))

    results = model.predict(
        source=str(SOURCE),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(OUTPUT),
        name="baseline_predictions",
        device="cpu",
        max_det=50,
    )

    print("Prediction complete.")
    print("Saved to:")
    print(
        OUTPUT / "baseline_predictions"
    )


if __name__ == "__main__":
    main()
