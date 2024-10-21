try:
    from ultralytics import YOLO
    import cv2
    from pathlib import Path
    from tqdm.rich import tqdm
    import pandas as pd
    from icecream import ic
    import yaml
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def main(
    model_path: Path,
    yolo_yaml_path: Path,
    out_path: Path,
    imgsz=640,
    iou=0.7,
    conf=0.5,
    only_original_images: bool = False,
):
    out_path.mkdir(exist_ok=True, parents=True)
    # Caricare un modello YOLO pre-addestrato
    model = YOLO(str(model_path))  # Puoi sostituire con il tuo modello personalizzato

    with open(str(yolo_yaml_path), "r") as f:
        data = yaml.safe_load(f)
    test_set_path = Path(data["path"]).joinpath(data["test"])
    images = [f for f in test_set_path.iterdir() if f.suffix.lower() == ".png"]
    for image in tqdm(images):
        char_list = "CNRBDF" if only_original_images else ""
        if all(char not in image.stem for char in char_list):
            # Fare inferenza su ogni frame
            results = model.predict(
                image,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
            )

            # Annotare i risultati sul frame
            annotated_frame = results[0].plot()

            cv2.imwrite(str(out_path.joinpath(f"{image.name}")), annotated_frame)

    print(f"Test set annotato salvato in: {out_path}")


def get_metrics(
    model_path: Path,
    yolo_yaml_path: Path,
    out_path: Path,
    imgsz: int = 640,
    multiple_confidence_thresholds: bool = False,
):
    confidence_thresholds = [0.30, 0.50, 0.70, 0.90]
    out_path.mkdir(exist_ok=True, parents=True)
    # Caricare un modello YOLO pre-addestrato
    model = YOLO(str(model_path))  # Puoi sostituire con il tuo modello personalizzato

    # Funzione per calcolare mAP50 e mAP50-95
    def calculate_map(model, conf=None):
        if conf is None:
            metrics = model.val(
                data=str(yolo_yaml_path), imgsz=imgsz, split="test", save_json=True
            )
        else:
            metrics = model.val(
                data=str(yolo_yaml_path),
                conf=conf,
                imgsz=imgsz,
                split="test",
                save_json=True,
            )
        attributi = dir(metrics)
        ic(attributi)

        classes = metrics.names

        confusion_matrix = metrics.confusion_matrix.matrix

        images_class_0 = int(np.sum(metrics.confusion_matrix.matrix[:, 0]))

        class_0_results = metrics.class_result(0)  # classe 0

        results_df = pd.DataFrame(
            {
                "Class": [f"{classes[0]}"],
                "Instances": [
                    images_class_0,
                ],
                "Precision": [
                    class_0_results[0],
                ],
                "Recall": [
                    class_0_results[1],
                ],
                "mAP50": class_0_results[2],
                "mAP50-95": [
                    class_0_results[3],
                ],
            }
        )

        curves_df_list = []
        for i in range(4):
            label_x = metrics.curves_results[i][2]
            label_y = metrics.curves_results[i][3]
            x = metrics.curves_results[i][0]
            y_class_0 = (metrics.curves_results[i][1])[0, :]
            df = pd.DataFrame(
                {
                    f"{label_x}": x,
                    f"{label_y}_{classes[0]}": y_class_0,
                }
            )
            curves_df_list.append(df)

        speed_df = pd.DataFrame({k: [v] for k, v in metrics.speed.items()})

        return results_df, curves_df_list, confusion_matrix, speed_df

    # Ciclo su varie soglie di confidenza
    df_names = ["PR_curve", "F1_curve", "P_curve", "R_curve"]
    if multiple_confidence_thresholds:
        for conf in confidence_thresholds:
            results_df, curves_df_list, confusion_matrix, speed_df = calculate_map(
                model, conf
            )

            results_df.to_csv(out_path.joinpath(f"{conf}_metrics.csv"), index=False)

            np.savetxt(
                out_path.joinpath(f"{conf}_confusion_matrix.txt"), confusion_matrix
            )

            speed_df.to_csv(out_path.joinpath(f"{conf}_speeds.csv"), index=False)

            for i in range(4):
                curves_df_list[i].to_csv(
                    out_path.joinpath(f"{conf}_{df_names[i]}.csv"), index=False
                )
    else:
        results_df, curves_df_list, confusion_matrix, speed_df = calculate_map(model)
        results_df.to_csv(out_path.joinpath(f"metrics.csv"), index=False)
        np.savetxt(out_path.joinpath(f"confusion_matrix.txt"), confusion_matrix)
        speed_df.to_csv(out_path.joinpath(f"speeds.csv"), index=False)
        for i in range(4):
            curves_df_list[i].to_csv(
                out_path.joinpath(f"{df_names[i]}.csv"), index=False
            )


def plot_results(out_path: Path):
    if not out_path.is_dir():
        raise Exception(f"{out_path} is not a directory!")
    out_path.joinpath("plots").mkdir(exist_ok=True, parents=True)

    targets = ["P", "R", "F1"]

    target_labels = {"P": "Precision", "R": "Recall", "F1": "F1"}

    for target in targets:
        df = pd.read_csv(out_path.joinpath(f"./{target}_curve.csv"))

        plt.style.use("seaborn-v0_8-paper")
        # sns.set_context("talk", font_scale=1.2)
        # sns.set_style("whitegrid")

        plt.figure(figsize=(7, 7))
        sns.lineplot(
            x="Confidence",
            y=f"{target_labels[target]}_Defect",
            data=df,
            label="Defect",
            color="blue",
            linewidth=1.5,
        )

        # Personalizzazione
        plt.xlabel("Confidence", fontsize=23)
        plt.ylabel(f"{target_labels[target]}", fontsize=23)
        if target == "P":
            legend = plt.legend(
                loc="lower right",
                fontsize=20,
                edgecolor="black",
                framealpha=1.0,
            )
        else:
            legend = plt.legend(
                loc="lower left",
                fontsize=20,
                edgecolor="black",
                framealpha=1.0,
            )
        legend.get_frame().set_linewidth(1)

        plt.grid(True, which="both", linestyle="--", linewidth=0.6)
        plt.xlim((-0.03, 1.03))
        plt.ylim(top=1.03)
        if target == "P":
            plt.ylim(bottom=0.27)
        else:
            plt.ylim(bottom=-0.03)

        plt.tight_layout()
        plt.tick_params(axis="both", which="major", labelsize=23)
        plt.savefig(
            out_path.joinpath("plots", f"{target}_curve.png"),
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    ic.disable()
    yolo_model = "yolov10n"
    model_path = Path(f"./best_model_{yolo_model}.pt")
    yolo_yaml_path = Path("./config/train.yaml")
    out_path = Path("./results")
    imgsz = 256
    main(model_path, yolo_yaml_path, out_path, imgsz=imgsz, iou=0.7, conf=0.5)
    out_path = out_path.joinpath("metrics")
    get_metrics(model_path, yolo_yaml_path, out_path, imgsz=imgsz)
    plot_results(out_path)
