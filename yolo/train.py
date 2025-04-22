try:
    from ultralytics import YOLO
    from pathlib import Path
    import time
    import shutil

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def main():
    model = YOLO("yolov10n.pt")

    try:
        shutil.rmtree(Path(__file__).parent.joinpath("runs"))
    except:
        pass

    start = time.time()
    results = model.train(
        data=str(Path(__file__).parent.joinpath("config", "cfg.yaml")),
        epochs=300,
        pretrained=True,
        # iou=0.5,
        visualize=True,
        patience=0,
        batch=32,  # 32
        imgsz=256,
        dropout=0.0,  # 0.0
        deterministic=True,
        seed=42,
    )  # train the model
    results = model.val()  # evaluate model performance on the validation set
    end = time.time()

    shutil.copy(
        Path(__file__).parent.joinpath("runs", "detect", "train", "weights", "best.pt"),
        Path(__file__).parent.joinpath("best_model_yolov10n.pt"),
    )

    print(f"Completed training in {(end - start) / 60:.3f} minutes")


if __name__ == "__main__":
    main()
