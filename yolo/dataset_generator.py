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
        if "color" in augmentation:
            aug_color = change_color(image)
            cv2.imwrite(str(image.with_stem(f"{image.stem}_C")), aug_color)
            shutil.copy(label, label.with_stem(f"{label.stem}_C"))
        if "negative" in augmentation:
            aug_negative = invert_colors(image)
            cv2.imwrite(str(image.with_stem(f"{image.stem}_N")), aug_negative)
            shutil.copy(label, label.with_stem(f"{label.stem}_N"))


def dataset_generator(yolo_data_path: Path, dpath: Path, split=0.9):
    print("Starting dataset creation ...")
    samples = [
        f
        for f in yolo_data_path.joinpath("images").iterdir()
        if f.suffix.lower() in IMAGES_SUFFIX
    ]

    dpath.joinpath("images", "train").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("images", "val").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("labels", "train").mkdir(exist_ok=True, parents=True)
    dpath.joinpath("labels", "val").mkdir(exist_ok=True, parents=True)

    k = int(round(len(samples) * split))

    train_samples = random.sample(samples, k)
    val_samples = [elem for elem in samples if elem not in train_samples]

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
            f"The number of images ({len(images)}) is different wrt to the number of labels ({len(labels)})!"
        )

    for image, label in zip(images, labels):
        if label.stat().st_size == 0 or check_admitted_labels(label, admitted_labels):
            os.remove(image)
            os.remove(label)
            ic(f"Removing {image}")
            ic(f"Removing {label}")


def delete_similar_images(
    images_folder_path: Path, labels_folder_path: Path, ssim_threshold: float
):
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
            os.remove(str(labels_folder_path.joinpath(item[n]).with_suffix(".txt")))
            ic(f"Removing file: {str(images_folder_path.joinpath(item[n]))}")


def check_unsupported_file_types(data_path: Path):
    files = [
        f
        for f in data_path.joinpath("images").iterdir()
        if (f.suffix.lower() not in IMAGES_SUFFIX)
    ]
    if len(files) > 0:
        raise Exception(
            f"Unsopported file types detected in {data_path.joinpath('images')}"
        )

    labels = [
        f
        for f in data_path.joinpath("labels").iterdir()
        if (f.suffix.lower() != ".txt")
    ]
    if len(labels) > 0:
        raise Exception(
            f"Unsopported file types detected in {data_path.joinpath('labels')}"
        )


@hydra.main(version_base="1.2", config_path="config", config_name="dataset")
def main(cfg):
    temp_folder = Path(cfg.package_path).joinpath("temp")
    shutil.copytree(Path(cfg.exported_data_path), temp_folder)
    check_unsupported_file_types(temp_folder)

    check_labels_consistency(temp_folder, list(cfg.admitted_labels))
    data_augmentation(temp_folder, list(cfg.augmentation))
    dataset_generator(temp_folder, Path(cfg.dataset_path), split=cfg.split)

    shutil.rmtree(temp_folder)
    shutil.rmtree(Path(cfg.package_path).joinpath("outputs"))


if __name__ == "__main__":
    ic.disable()
    main()
