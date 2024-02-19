try:
    import cv2
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm
    from scipy.fftpack import fft2, fftshift
    from icecream import ic

except Exception as e:
    print(f"Some module are missing for {__file__}: {e}\n")


class Features:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

    def extract_shape_features(
        image: Path | np.ndarray,
        dest_path: Path | None = None,
        grayscale: bool = False,
        min_area: int = 1,
    ) -> dict:
        # Load image as grayscale
        if isinstance(image, Path):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE if grayscale else -1)
        else:
            img = image.copy()

        # Threshold image to create binary mask
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

        # Filter contours based on area
        contours = [
            contour for contour in contours if cv2.contourArea(contour) > min_area
        ]

        if len(contours) == 0:
            return None
        # Get largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter**2)
        solidity = (
            cv2.contourArea(largest_contour)
            / cv2.convexHull(largest_contour, returnPoints=False).size
        )

        compactness = perimeter**2 / area

        _, (diam_x, diam_y), _ = cv2.minAreaRect(largest_contour)
        feret_diameter = max(diam_x, diam_y)

        if diam_x < diam_y:
            diam_x, diam_y = diam_y, diam_x

        eccentricity = np.sqrt(1 - (diam_y / diam_x) ** 2)

        if dest_path is not None:
            # Draw largest contour on input image
            img_with_contour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_with_contour, [largest_contour], -1, (0, 0, 255), 3)
            cv2.imwrite(str(dest_path), img_with_contour)

        # Create dictionary of shape features
        num_pixels = img.shape[0] * img.shape[1]
        shape_features = {
            "area": area,
            "num_pixels": num_pixels,
            "perimeter": perimeter,
            "circularity": circularity,
            "solidity": solidity,
            "compactness": compactness,
            "feret_diameter": feret_diameter,
            "eccentricity": eccentricity,
        }

        return shape_features

    def extract_edge_features(
        image: Path | np.ndarray,
        grayscale: bool = False,
        min_area: int = 1,
    ) -> dict:
        # Load image as grayscale
        if isinstance(image, Path):
            img = cv2.imread(str(image), 0 if grayscale else -1)
        else:
            img = image.copy()

        # Apply Canny edge detection algorithm
        edges = cv2.Canny(img, 100, 200)

        # Compute edge features
        num_edges = np.sum(edges == 255)
        edge_density = num_edges / (img.shape[0] * img.shape[1])

        # Find contours in binary mask
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        contours = [
            contour for contour in contours if cv2.contourArea(contour) > min_area
        ]

        if len(contours) == 0:
            return None

        # Compute mean and standard deviation of contour lengths
        contour_lengths = [cv2.arcLength(contour, True) for contour in contours]
        if len(contour_lengths) > 0:
            mean_contour_length = np.mean(contour_lengths)
            std_contour_length = np.std(contour_lengths)
        else:
            mean_contour_length = 0
            std_contour_length = 0

        edge_features = {
            "number_of_edges": num_edges,
            "edge_density": edge_density,
            # "mean_length_of_edges": mean_contour_length,
            # "std_length_of_edges": std_contour_length,
        }

        return edge_features


if __name__ == "__main__":
    pass
