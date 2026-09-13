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

DATA = (
    ROOT
    / "data"
    / "yolo_road_damage"
    / "data.yaml"
)

OUTPUT = ROOT / "models" / "evaluation"


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL}")

    if not DATA.exists():
        raise FileNotFoundError(f"Dataset config not found:\n{DATA}")

    model = YOLO(str(MODEL))

    print("=" * 65)
    print("NagarSetu - BASELINE EVALUATION")
    print("=" * 65)

    metrics = model.val(
        data=str(DATA),
        split="val",
        imgsz=640,
        batch=8,
        plots=True,
        project=str(OUTPUT),
        name="baseline_eval",
        device="cpu",
    )

    print("\nRESULTS")
    print("-" * 65)

    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")

    print("\nPer-class mAP50:")

    names = model.names

    for class_id, value in enumerate(metrics.box.ap50):
        print(
            f"{class_id} - "
            f"{names[class_id]}: "
            f"{value:.4f}"
        )

    print("\nEvaluation files:")
    print(OUTPUT / "baseline_eval")


if __name__ == "__main__":
    main()
