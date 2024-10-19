try:
    import cv2
    from skimage.metrics import structural_similarity as ssim
    from pathlib import Path
    from tqdm.rich import tqdm
    import os
    from icecream import ic
    import hydra
    import shutil
    from lib.lib_data_augmentation import *
    import random
    import pandas as pd

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")

IMAGES_SUFFIX = (".jpg", ".png", ".jpeg")


# Funzione per calcolare la similarità strutturale tra due immagini
def calculate_ssim(image_path1: Path, image_path2: Path):
    # Converti le immagini in scala di grigi
    img1 = cv2.imread(str(image_path1))
    img2 = cv2.imread(str(image_path2))
    gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calcola la similarità strutturale (SSIM) tra le due immagini
    if gray_image1.shape == gray_image2.shape:
        similarity_index, _ = ssim(gray_image1, gray_image2, full=True)
        return similarity_index
    else:
        return -1


def merge_duplicate_images(image_pairs):
    image_dict = {}

    for pair in image_pairs:
        merged = False
        group = set(pair)
        for img in pair:
            if img in image_dict:
                group |= image_dict[img]
                merged = True

        for img in group:
            image_dict[img] = group

    unique_groups = list(set(tuple(sorted(group)) for group in image_dict.values()))

    return unique_groups


def estrai_frames_da_video(video_path, output_folder, intervallo=20):
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Leggi il video
    cap = cv2.VideoCapture(video_path)

    # Controlla se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'apertura del file video.")
        return

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # Ottieni i frame al secondo del video
    frame_interval = int(fps * intervallo)  # Calcola l'intervallo di frame

    while True:
        ret, frame = cap.read()  # Leggi un frame
        if not ret:
            break  # Se non ci sono più frame, esci dal ciclo

        # Salva il frame solo se è il frame corretto da campionare
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(
                output_folder, f"frame_2_{frame_count // frame_interval:04d}.png"
            )
            cv2.imwrite(frame_name, frame)

        frame_count += 1

    cap.release()  # Rilascia il video quando hai finito
    print(
        f"Estrazione completata. Sono stati salvati i frame ogni {intervallo} secondi nella cartella {output_folder}."
    )


def data_augmentation(yolo_data_path: Path, augmentation: list):
    print("Starting data augmentation ...")
    images = [
        f
        for f in yolo_data_path.joinpath("images").iterdir()
        if f.suffix.lower() in IMAGES_SUFFIX
    ]
    images = sorted(images, key=lambda x: str(x).lower())

    labels = [
        f
        for f in yolo_data_path.joinpath("labels").iterdir()
        if f.suffix.lower() == ".txt"
    ]
    labels = sorted(labels, key=lambda x: str(x).lower())

    for image, label in zip(images, labels):
        ic(image)
        if "color" in augmentation:
            aug_color = change_color(image)
            cv2.imwrite(str(image.with_stem(f"{image.stem}_C")), aug_color)
            shutil.copy(label, label.with_stem(f"{label.stem}_C"))
        if "negative" in augmentation:
            aug_negative = invert_colors(image)
            cv2.imwrite(str(image.with_stem(f"{image.stem}_N")), aug_negative)
            shutil.copy(label, label.with_stem(f"{label.stem}_N"))
        if "noise" in augmentation:
            aug_noisy = apply_noise(image)
            cv2.imwrite(str(image.with_stem(f"{image.stem}_R")), aug_noisy)
            shutil.copy(label, label.with_stem(f"{label.stem}_R"))
        if "brightness" in augmentation:
            aug_bright = adjust_brightness(image, brightness_factor=70)
            cv2.imwrite(str(image.with_stem(f"{image.stem}_B")), aug_bright)
            shutil.copy(label, label.with_stem(f"{label.stem}_B"))

            aug_dark = adjust_brightness(image, brightness_factor=-70)
            cv2.imwrite(str(image.with_stem(f"{image.stem}_D")), aug_dark)
            shutil.copy(label, label.with_stem(f"{label.stem}_D"))
        if "rotate" in augmentation:
            # rotate_yolo_labels(
            #     label,
            #     label.with_stem(f"{label.stem}_R90"),
            #     image,
            #     image.with_stem(f"{image.stem}_R90"),
            #     angle=90,
            # )
            rotate_yolo_labels(
                label,
                label.with_stem(f"{label.stem}_R180"),
                image,
                image.with_stem(f"{image.stem}_R180"),
                angle=180,
            )
            # rotate_yolo_labels(
            #     label,
            #     label.with_stem(f"{label.stem}_R270"),
            #     image,
            #     image.with_stem(f"{image.stem}_R270"),
            #     angle=270,
            # )
        if "flip" in augmentation:
            flip_labels_yolo(
                label,
                label.with_stem(f"{label.stem}_FX"),
                image,
                image.with_stem(f"{image.stem}_FX"),
                flip="x",
            )
            flip_labels_yolo(
                label,
                label.with_stem(f"{label.stem}_FY"),
                image,
                image.with_stem(f"{image.stem}_FY"),
                flip="y",
            )
            flip_labels_yolo(
                label,
                label.with_stem(f"{label.stem}_FXY"),
                image,
                image.with_stem(f"{image.stem}_FXY"),
                flip="xy",
            )


def dataset_generator(yolo_data_path: Path, dpath: Path, split=0.7):
    print("Starting dataset creation ...")
    samples = [
        f
        for f in yolo_data_path.joinpath("images").iterdir()
        if f.suffix.lower() in IMAGES_SUFFIX
    ]

    dpath.joinpath("images", "train").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("images", "val").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("images", "test").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("labels", "train").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("labels", "val").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("labels", "test").mkdir(exist_ok=True, parents=True)

    train_split = split
    val_split = (1 - train_split) / 2

    k_train = int(round(len(samples) * train_split))
    k_val = int(round(len(samples) * val_split))

    train_samples = random.sample(samples, k_train)

    remaining_samples = [elem for elem in samples if elem not in train_samples]
    val_samples = random.sample(remaining_samples, k_val)
    test_samples = [elem for elem in remaining_samples if elem not in val_samples]

    for sample in tqdm(train_samples):
        shutil.copy(sample, dpath.joinpath("images", "train", sample.name))
        shutil.copy(
            yolo_data_path.joinpath("labels", f"{sample.stem}.txt"),
            dpath.joinpath("labels", "train", f"{sample.stem}.txt"),
        )

    train_samples = sorted(train_samples, key=lambda x: str(x).lower())
    df = pd.DataFrame({"file_name": [f.stem for f in train_samples]})
    df.to_csv(dpath.joinpath("train.csv"), index=False)

    for sample in tqdm(val_samples):
        shutil.copy(sample, dpath.joinpath("images", "val", sample.name))
        shutil.copy(
            yolo_data_path.joinpath("labels", f"{sample.stem}.txt"),
            dpath.joinpath("labels", "val", f"{sample.stem}.txt"),
        )

    val_samples = sorted(val_samples, key=lambda x: str(x).lower())
    df = pd.DataFrame({"file_name": [f.stem for f in val_samples]})
    df.to_csv(dpath.joinpath("val.csv"), index=False)

    for sample in tqdm(test_samples):
        shutil.copy(sample, dpath.joinpath("images", "test", sample.name))
        shutil.copy(
            yolo_data_path.joinpath("labels", f"{sample.stem}.txt"),
            dpath.joinpath("labels", "test", f"{sample.stem}.txt"),
        )

    test_samples = sorted(test_samples, key=lambda x: str(x).lower())
    df = pd.DataFrame({"file_name": [f.stem for f in test_samples]})
    df.to_csv(dpath.joinpath("test.csv"), index=False)


def delete_similar_images(images_folder_path: Path, ssim_threshold: float):
    images = [
        f for f in images_folder_path.iterdir() if f.suffix.lower() in IMAGES_SUFFIX
    ]
    images = sorted(images, key=lambda x: str(x).lower())

    similar_images = []
    for i in tqdm(range(len(images) - 1)):
        similarity = calculate_ssim(images[i], images[i + 1])
        if similarity >= ssim_threshold:
            similar_images.append((images[i].name, images[i + 1].name))

    similar_images = merge_duplicate_images(similar_images)
    ic(similar_images)
    for item in similar_images:
        for n in range(len(item) - 1):
            os.remove(str(images_folder_path.joinpath(item[n])))
            ic(f"Removing file: {str(images_folder_path.joinpath(item[n]))}")


def check_unsupported_file_types(data_path: Path):
    files = [
        f
        for f in data_path.joinpath("images").iterdir()
        if f.suffix.lower() not in IMAGES_SUFFIX
    ]
    if len(files) > 0:
        raise Exception(f"Unsopported file types detected in {data_path.joinpath(dir)}")

    files = [
        f for f in data_path.joinpath("labels").iterdir() if f.suffix.lower() != ".txt"
    ]
    if len(files) > 0:
        raise Exception(f"Unsopported file types detected in {data_path.joinpath(dir)}")


def check_admitted_labels(label_path: Path, admitted_labels: list):
    with open(str(label_path), "r") as file:
        for row in file:
            elements = row.split()

            if len(elements) > 0 and str(elements[0]) not in admitted_labels:
                return True

    return False


def check_labels_consistency(yolo_data_path: Path, admitted_labels: list):
    images = [
        f
        for f in yolo_data_path.joinpath("images").iterdir()
        if f.suffix.lower() in IMAGES_SUFFIX
    ]
    images = sorted(images, key=lambda x: str(x).lower())

    labels = [
        f
        for f in yolo_data_path.joinpath("labels").iterdir()
        if f.suffix.lower() == ".txt"
    ]
    labels = sorted(labels, key=lambda x: str(x).lower())

    if len(images) != len(labels):
        raise Exception(
            "The number of images is different wrt to the number oof labels!"
        )

    for image, label in zip(images, labels):
        if label.stat().st_size == 0 or check_admitted_labels(label, admitted_labels):
            os.remove(image)
            os.remove(label)
            ic(f"Removing {image}")
            ic(f"Removing {label}")


@hydra.main(version_base="1.2", config_path="config", config_name="dataset")
def main(cfg):
    temp_folder = Path(cfg.package_path).joinpath("temp")
    shutil.copytree(Path(cfg.exported_data_path), temp_folder)
    check_unsupported_file_types(temp_folder)

    check_labels_consistency(temp_folder, list(cfg.admitted_labels))
    # delete_similar_images(
    #     temp_folder.joinpath("images"),
    #     temp_folder.joinpath("labels"),
    #     cfg.ssim_threshold,
    # )
    data_augmentation(temp_folder, list(cfg.augmentation))
    dataset_generator(temp_folder, Path(cfg.dataset_path), split=cfg.split)

    shutil.rmtree(temp_folder)
    shutil.rmtree(Path(cfg.package_path).joinpath("outputs"))


if __name__ == "__main__":
    ic.disable()
    main()
    # estrai_frames_da_video(
    #     "/home/tom/git_workspace/yolov10/data/video/Video 3 - Muto.mp4",
    #     "/home/tom/git_workspace/yolov10/data/frames",
    #     intervallo=1,
    # )
