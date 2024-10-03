try:
    import cv2
    from pathlib import Path
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from scipy.stats import skew, kurtosis
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
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

        edge_features = {
            "number_of_edges": num_edges,
            "edge_density": edge_density,
        }

        return edge_features

    def extract_texture_features(image: Path | np.ndarray) -> dict:
        # Load image
        if isinstance(image, Path):
            img = cv2.imread(str(image), 0)
        else:
            img = image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute GLCM matrix
        glcm = graycomatrix(
            img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
        )

        # Compute texture features
        contrast = graycoprops(glcm, "contrast")[0][0]
        homogeneity = graycoprops(glcm, "homogeneity")[0][0]
        energy = graycoprops(glcm, "energy")[0][0]
        correlation = graycoprops(glcm, "correlation")[0][0]

        texture_features = {
            "GLCM_contrast": contrast,
            "GLCM_homogeneity": homogeneity,
            "GLCM_energy": energy,
            "GLCM_correlation": correlation,
        }

        return texture_features
    def extract_frequency_features(image: Path | np.ndarray) -> dict:
        # Load image as grayscale
        if isinstance(image, Path):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        else:
            img = image.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Fourier Transform
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # Avoid division by zero by adding a small constant
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)

        # Compute frequency features
        mean_freq = np.mean(magnitude_spectrum)
        std_freq = np.std(magnitude_spectrum)
        max_freq = np.max(magnitude_spectrum)
        min_freq = np.min(magnitude_spectrum)
        median_freq = np.median(magnitude_spectrum)
        energy = np.sum(np.abs(fshift)**2)
        
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2  # center
 
        histogram, _ = np.histogram(magnitude_spectrum, bins=256, range=(0, 256), density=True)
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))  # Adding a small value to avoid log(0)
        
    
        skewness = skew(magnitude_spectrum.flatten())
        kurt = kurtosis(magnitude_spectrum.flatten())
        
        low_freq_energy = np.sum(magnitude_spectrum[:crow, :ccol])
        high_freq_energy = np.sum(magnitude_spectrum[crow:, ccol:])
        frequency_contrast = high_freq_energy - low_freq_energy

        frequency_features = {
            "mean_frequency": mean_freq,
            "std_frequency": std_freq,
            "max_frequency": max_freq,
            "min_frequency": min_freq,
            "median_frequency": median_freq,
            "energy": energy,
            "entropy": entropy,
            "skewness": skewness,
            "kurtosis": kurt,
            "low_frequency_energy": low_freq_energy,
            "high_frequency_energy": high_freq_energy,
            "frequency_contrast": frequency_contrast,
        }

        return frequency_features



if __name__ == "__main__":
    pass
    # from tqdm.rich import tqdm

    # crops_path = Path("/home/tom/git_workspace/GrapheNetDefectDetector/data/crops")
    # images = [
    #     f for f in crops_path.iterdir() if f.suffix.lower() in Features.IMAGE_EXTENSIONS
    # ]

    # for image in tqdm(images):

    #     shape_features = Features.extract_shape_features(
    #         image,
    #         dest_path=Path(
    #             "/home/tom/git_workspace/GrapheNetDefectDetector/data/test_area"
    #         ).joinpath(f"{image.stem}.png"),
    #         grayscale=True,
    #         min_area=1,
    #     )
