try:
    import cv2
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def apply_blur(img_path: Path, kernel_size=(5, 5), sigma=2.0):
    image = cv2.imread(str(img_path))
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image


def change_color(img_path: Path):
    image = cv2.imread(str(img_path))
    new_image = image.copy()
    new_image[:, :, 0] = image[:, :, 2]
    new_image[:, :, 2] = image[:, :, 0]
    return new_image


def invert_colors(img_path: Path):
    image = cv2.imread(str(img_path))
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def apply_noise(img_path: Path, noise_probability: float = 0.08):
    image = cv2.imread(str(img_path), 0)
    noise = np.random.rand(*image.shape[:2])

    noisy_image = np.copy(image)
    noisy_image[noise < noise_probability / 2] = 0
    noisy_image[noise > 1 - noise_probability / 2] = 255
    return noisy_image


def adjust_brightness(img_path: Path, brightness_factor: float = 70):
    image = cv2.imread(str(img_path))

    if brightness_factor >= 0:
        M = np.ones(image.shape, dtype="uint8") * brightness_factor
        bright_image = cv2.add(image, M)
    else:
        brightness_factor *= -1
        M = np.ones(image.shape, dtype="uint8") * brightness_factor
        bright_image = cv2.subtract(image, M)
    return bright_image


def rotate_yolo_labels(
    yolo_label_path: Path,
    out_label_path: Path,
    img_path: Path,
    out_img_path: Path,
    angle: int = 90,
):
    if angle not in [90, 180, 270]:
        raise ValueError("Accepted value for angle: 90,180,270")

    image = cv2.imread(str(img_path))
    img_height, img_width, _ = image.shape

    if angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(out_img_path), rotated_image)

    with open(str(yolo_label_path), "r") as f:
        lines = f.readlines()

    rotated_lines = []

    for line in lines:
        class_idx, x_center_norm, y_center_norm, width_norm, height_norm = map(
            float, line.split()
        )
        class_idx = int(class_idx)

        # Convert normalized center coordinates to pixel coordinates
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height

        # Compute new pixel coordinates for rotated bounding box
        if angle == 180:
            x_new = img_width - x_center
            y_new = img_height - y_center
        elif angle == 270:
            x_new = y_center
            y_new = img_width - x_center
        elif angle == 90:
            x_new = img_height - y_center
            y_new = x_center

        # Convert new pixel coordinates back to normalized coordinates
        if angle == 180:
            x_new_norm = x_new / img_width
            y_new_norm = y_new / img_height
        elif angle == 90 or angle == 270:
            x_new_norm = x_new / img_height
            y_new_norm = y_new / img_width

        # Width and height remain unchanged
        if angle == 180:
            width_new_norm = width_norm
            height_new_norm = height_norm
        elif angle == 90 or angle == 270:
            width_new_norm = height_norm
            height_new_norm = width_norm

        # Append rotated annotation to list of strings
        rotated_line = f"{class_idx} {x_new_norm:.6f} {y_new_norm:.6f} {width_new_norm:.6f} {height_new_norm:.6f}\n"
        rotated_lines.append(rotated_line)

    with open(str(out_label_path), "w") as f:
        f.writelines(rotated_lines)


def flip_labels_yolo(
    label_path: Path,
    out_label_path: Path,
    image_path: Path,
    out_image_path: Path,
    flip: str = "x",
):
    if flip not in ["x", "y", "xy"]:
        raise ValueError("Accepted values: 'x', 'y' and 'xy'")
    # Apri il file delle etichette in lettura
    with open(str(label_path), "r") as file:
        lines = file.readlines()

    image = cv2.imread(str(image_path))
    if flip == "y":
        flipped_image = cv2.flip(image, 1)
    elif flip == "x":
        flipped_image = cv2.flip(image, 0)
    elif flip == "xy":
        flipped_image = cv2.flip(image, -1)
    cv2.imwrite(str(out_image_path), flipped_image)

    # Lista per salvare le nuove etichette
    new_lines = []

    # Per ogni riga nel file
    for line in lines:
        # Dividi i valori (classe, x_center, y_center, width, height)
        values = line.strip().split()
        class_id = values[0]
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        # Applica la trasformazione al centro x (flip lungo l'asse Y)
        if flip == "y":
            new_x_center = 1 - x_center
            new_y_center = y_center
        elif flip == "x":
            new_x_center = x_center
            new_y_center = 1 - y_center
        elif flip == "xy":
            new_x_center = 1 - x_center
            new_y_center = 1 - y_center

        # Crea la nuova riga con le coordinate aggiornate
        new_line = f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {width:.6f} {height:.6f}\n"
        new_lines.append(new_line)

    # Scrivi le nuove etichette in un nuovo file o sovrascrivi il file esistente
    with open(str(out_label_path), "w") as file:
        file.writelines(new_lines)


def plot_image_with_labels(image_path, labels_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Converti da BGR a RGB per matplotlib

    # Ottieni le dimensioni dell'immagine
    height, width, _ = image.shape

    # Inizializza la figura
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Leggi le etichette
    with open(labels_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Parse le etichette
        class_id, center_x, center_y, box_width, box_height = map(
            float, line.strip().split()
        )

        # Calcola le coordinate della bounding box
        x_min = int((center_x - box_width / 2) * width)
        x_max = int((center_x + box_width / 2) * width)
        y_min = int((center_y - box_height / 2) * height)
        y_max = int((center_y + box_height / 2) * height)

        # Disegna la bounding box
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        plt.gca().add_patch(rect)

        # Aggiungi il label della classe
        plt.text(
            x_min,
            y_min,
            f"Class {int(class_id)}",
            color="red",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    plt.axis("off")  # Nascondi gli assi
    plt.show()
