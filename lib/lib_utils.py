# imports
try:
    import numpy as np
    import shutil
    from tqdm import tqdm
    import random
    from PIL import Image
    from pathlib import Path
    import math
    from chemfiles import Trajectory
    from PIL import Image, ImageDraw
    import cv2
    import torch
    from ultralytics import YOLO
    from icecream import ic

except Exception as e:
    print("Some module are missing {}".format(e))


class Utils:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

    @staticmethod
    def generate_bonds_png(
        spath: Path,
        dpath: Path,
        max_dim: list,
        multiplier: int = 3,
    ):

        with Trajectory(str(spath)) as trajectory:
            mol = trajectory.read()

        resolution = round(
            multiplier * (5 + np.max([np.abs(max_dim[0]), np.abs(max_dim[1])]))
        )

        B = Image.new("RGB", (resolution, resolution))
        B_ = ImageDraw.Draw(B)

        mol.guess_bonds()
        if mol.topology.bonds_count() == 0:
            print(f"No bonds guessed for {spath.stem}\n")
        bonds = mol.topology.bonds

        for i in range(len(bonds)):
            x_1 = int(round(mol.positions[bonds[i][0]][0] * multiplier))
            y_1 = int(round(mol.positions[bonds[i][0]][1] * multiplier))
            x_2 = int(round(mol.positions[bonds[i][1]][0] * multiplier))
            y_2 = int(round(mol.positions[bonds[i][1]][1] * multiplier))
            line = [(x_1, y_1), (x_2, y_2)]
            first_atom = mol.atoms[bonds[i][0]].name
            second_atom = mol.atoms[bonds[i][1]].name
            color = Utils.find_bound_type(first_atom, second_atom)
            B_.line(line, fill=color, width=0)

        B = Utils.crop_image(B)
        B.save(str(dpath.joinpath(f"{spath.stem}.png")))

    @staticmethod
    def find_bound_type(first_atom: str, second_atom: str) -> str:
        if (first_atom == "C" and second_atom == "C") or (
            second_atom == "C" and first_atom == "C"
        ):
            return "red"
        elif (first_atom == "C" and second_atom == "O") or (
            second_atom == "O" and first_atom == "C"
        ):
            return "blue"
        elif (first_atom == "O" and second_atom == "H") or (
            second_atom == "H" and first_atom == "O"
        ):
            return "white"
        elif (first_atom == "C" and second_atom == "H") or (
            second_atom == "H" and first_atom == "C"
        ):
            return "yellow"

    @staticmethod
    def crop_image(image: Image, name: str = None, dpath: Path = None) -> Image:

        image_data = np.asarray(image)
        if len(image_data.shape) == 2:
            image_data_bw = image_data
        else:
            image_data_bw = image_data.max(axis=2)
        non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
        cropBox = (
            min(non_empty_rows),
            max(non_empty_rows),
            min(non_empty_columns),
            max(non_empty_columns),
        )

        if len(image_data.shape) == 2:
            image_data_new = image_data[
                cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1
            ]
        else:
            image_data_new = image_data[
                cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :
            ]

        new_image = Image.fromarray(image_data_new)
        if dpath is not None:
            new_image.save(dpath.joinpath(name))

        return new_image

    @staticmethod
    def split_dataset(spath: Path, dpath: Path, split: float = 0.8):
        train_path = dpath.joinpath("train")
        train_path.mkdir(exist_ok=True, parents=True)
        test_path = dpath.joinpath("test")
        test_path.mkdir(exist_ok=True, parents=True)

        labels = [f for f in spath.iterdir() if f.suffix.lower() == ".txt"]
        train_labels = random.sample(labels, int(len(labels) * split))
        test_labels = list(set(labels) - set(train_labels))

        for label in tqdm(train_labels):
            shutil.copy(label, train_path.joinpath(label.name))
            shutil.copy(
                label.with_suffix(".png"), train_path.joinpath(f"{label.stem}.png")
            )

        for label in tqdm(test_labels):
            shutil.copy(label, test_path.joinpath(label.name))
            shutil.copy(
                label.with_suffix(".png"), test_path.joinpath(f"{label.stem}.png")
            )

    @staticmethod
    def from_xyz_to_png(
        spath: Path,
        dpath: Path,
        max_dim: list,
        items: int = None,
        multiplier: int = 6,
    ):
        if dpath.is_dir():
            print(f"WARNING: the directory {dpath} already exists!")
            return
        else:
            dpath.mkdir(exist_ok=True, parents=True)

        files = [f for f in spath.iterdir() if f.suffix.lower() == ".xyz"]
        if items is None:
            items = len(files)

        pbar = tqdm(total=len(files) if items > len(files) else items)
        for i, file in enumerate(files):
            if i >= items:
                break
            Utils.generate_bonds_png(file, dpath, max_dim, multiplier)
            pbar.update(1)
        pbar.close()

    @staticmethod
    def generate_yolo_crops(
        spath: Path,
        dpath: Path,
        yolo_ckpt_path: Path,
        binary_mask: bool = False,
        device: str = "cuda",
        verbose: bool = False,
        confidence: float = 0.75,
    ):

        if dpath.is_dir():
            print(f"WARNING: the directory {dpath} already exists!")
            return
        else:
            dpath.mkdir(exist_ok=True, parents=True)

        model = YOLO(
            str(yolo_ckpt_path),
            task="detect",
        ).to(torch.device(device))

        images = [
            f for f in spath.iterdir() if f.suffix.lower() in Utils.IMAGE_EXTENSIONS
        ]

        for image in tqdm(images):

            results = model(str(image), verbose=verbose)
            boxes = results[0].boxes

            for i, box in enumerate(boxes):
                if box.conf < confidence:
                    continue
                array = torch.Tensor(box.xyxy).cpu().numpy()

                x1 = math.floor(array[0, 0])
                y1 = math.floor(array[0, 1])
                x2 = math.ceil(array[0, 2])
                y2 = math.ceil(array[0, 3])

                img = cv2.imread(str(image))

                cropped_image = img[y1:y2, x1:x2]

                if binary_mask:
                    defect1_grey = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                    # 2) apply binary thresholding, porto tutti i pixel significativi a 255 di bianco
                    ret, thresh = cv2.threshold(defect1_grey, 0, 255, cv2.THRESH_BINARY)

                    # 3) detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
                    contours, hierarchy = cv2.findContours(
                        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
                    )

                    # draw contours on the original image
                    image_copy = cropped_image.copy()
                    cv2.drawContours(
                        image=image_copy,
                        contours=contours,
                        contourIdx=-1,
                        color=(255, 255, 255),
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )

                    # Applica la soglia
                    ret, cropped_image = cv2.threshold(
                        image_copy, 245, 255, cv2.THRESH_BINARY_INV
                    )

                cv2.imwrite(
                    str(dpath.joinpath(f"{image.stem}_crop_{i}.png")),
                    cropped_image,
                )

    @staticmethod
    def read_from_xyz_file(file_path: Path):
        """Read xyz files and return lists of x,y,z coordinates and atoms"""

        X = []
        Y = []
        Z = []
        atoms = []

        with open(str(file_path), "r") as f:
            num_atom = int(next(f))
            next(f)  # ignore the comment

            for _ in range(num_atom):
                l = next(f).split()
                if len(l) == 4 or len(l) == 5:
                    X.append(float(l[1]))
                    Y.append(float(l[2]))
                    Z.append(float(l[3]))
                    atoms.append(str(l[0]))

        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        return atoms, X, Y, Z

    @staticmethod
    def check_x_interface_num_atoms(
        file_path: Path,
        num_atoms: int = 32,
        delta: float = 1.42,
    ):
        atoms, X, Y, Z = Utils.read_from_xyz_file(file_path)

        y_up = [y for y in Y if max(Y) - delta <= y <= max(Y)]
        y_down = [y for y in Y if min(Y) <= y <= min(Y) + delta]

        if len(y_up) == len(y_down) == num_atoms:
            return True
        else:
            ic(
                f"Warning, {file_path.name} failed check interface atom counts ({len(y_up)}!={len(y_down)}!={num_atoms}) and will be deleted!"
            )
            return False


if __name__ == "__main__":
    pass
    # import pandas as pd

    # ic.disable()

    # dataset = Path("/home/tom/git_workspace/GrapheNetDefectDetector/data/xyz_files")

    # files = [f for f in dataset.iterdir() if f.suffix.lower() == ".xyz"]

    # brokens = []
    # for file in tqdm(files):
    #     if not Utils.check_x_interface_num_atoms(file):
    #         brokens.append(file.stem)

    # df = pd.DataFrame(brokens, columns=["file_name"])
    # df.to_csv(dataset.joinpath("brokens.csv"), index=False)
