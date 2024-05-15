try:
    import cv2
    from pathlib import Path

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
