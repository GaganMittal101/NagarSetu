from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# NagarSetu
# Road Damage Detection - Baseline Training
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    ROOT
    / "data"
    / "yolo_road_damage"
    / "data.yaml"
)

RUNS_DIR = (
    ROOT
    / "models"
    / "runs"
)


def main():
    print("=" * 70)
    print("NagarSetu - ROAD DAMAGE DETECTOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset check
    # --------------------------------------------------------
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found:\n{DATA_YAML}"
        )

    # --------------------------------------------------------
    # Hardware check
    # --------------------------------------------------------
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device = 0

        print("\nCUDA: AVAILABLE")
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    else:
        device = "cpu"

        print("\nCUDA: NOT AVAILABLE")
        print("Training will use CPU.")

    print("\nDataset:")
    print(DATA_YAML)

    # --------------------------------------------------------
    # Load pretrained detector
    # --------------------------------------------------------
    print("\nLoading pretrained YOLO model...")

    model = YOLO("yolo26n.pt")

    # --------------------------------------------------------
    # Baseline training
    # --------------------------------------------------------
    print("\nStarting baseline training...")

    results = model.train(
        data=str(DATA_YAML),

        epochs=30,
        imgsz=640,
        batch=8,

        seed=42,
        workers=2,

        device=device,

        project=str(RUNS_DIR),
        name="NagarSetu_road_damage_baseline",

        save=True,
        plots=True,

        patience=8,
        cache=False,

        verbose=True,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nBest model:")
    print(
        RUNS_DIR
        / "NagarSetu_road_damage_baseline"
        / "weights"
        / "best.pt"
    )

    print("\nLast model:")
    print(
        RUNS_DIR
        / "NagarSetu_road_damage_baseline"
        / "weights"
        / "last.pt"
    )


if __name__ == "__main__":
    main()
