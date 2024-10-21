try:
    from ultralytics import YOLO
    from pathlib import Path
    import time
    import shutil

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def main():
    # yolo_model = "yolo11n.pt"
    yolo_model = "yolov10n.pt"
    model = YOLO(yolo_model)

    if Path(__file__).parent.joinpath("runs").is_dir():
        shutil.rmtree(Path(__file__).parent.joinpath("runs"))

    training_params = {
        "data": str(
            Path(__file__).parent.joinpath("config", "train.yaml")
        ),  # File YAML del dataset
        "epochs": 300,  # Numero di epoche
        "batch": 32,  # Dimensione del batch
        "imgsz": 256,  # Dimensione delle immagini (640x640)
        # "optimizer": "Adam",  # Ottimizzatore: puoi usare 'Adam', 'SGD', 'RMSProp', ecc.
        # "lr0": 0.001,  # Learning rate iniziale
        # "lrf": 0.01,  # Learning rate finale (scheduler)
        # "weight_decay": 0.0005,  # Peso per la penalizzazione L2 (Weight Decay)
        # "momentum": 0.937,  # Momento (per SGD)
        "workers": 6,  # Numero di processi paralleli per il caricamento dei dati
        "patience": 25,  # Numero di epoche di pazienza per l'early stopping
        "verbose": True,  # Stampa i dettagli del training
        "seed": 42,  # Imposta un seed per la riproducibilità
        "deterministic": False,
        "visualize": True,
        "pretrained": False,
    }

    start = time.time()
    results = model.train(**training_params)  # train the model
    metrics = model.val(split="test", save_json=True, plots=True)
    print(metrics.box.map)
    end = time.time()

    shutil.copy(
        Path(__file__).parent.joinpath("runs", "detect", "train", "weights", "best.pt"),
        Path(__file__).parent.joinpath(f"best_model_{yolo_model}"),
    )

    print(f"Completed training in {(end-start)/60:.3f} minutes")


if __name__ == "__main__":
    main()
