# Data-Driven Analysis and Generation of Defective Graphene Nanoflakes for Property Prediction and Device Optimization
<!-- Ho fatto il tuning dei parametri solo per il target total_energy, bisognerebbe farlo anche per energy_per_atom o qualche altro target che vogliamo studiare.

attualmente il file ``tuner.py``, ha come modello al suo interno solo il xgBoost  e salva tutto in ``optuna.log``. Per usare il tuner ho salvato i dati di training normalizzati con numpy e si chiamano
``X_norm.npy`` e ``Y_norm.npy``.

Si potrebbe eliminare tutta la parte sotto il fit che è la roba vecchia del tesista. -->

This repo is about using computer vision techniques (and in particular convolutional neural networks) in order to extract geometrical and frequency features from images and use such features with classical ML techniques to predict the electronic properties of defected graphene flakes. The framework take care to create the dataset starting from a folder of `.xyz` files and a `.csv` file containing the target properties, and uses YOLO for defect detection and XGBoost for target prediction.

## Project Structure
   ```
   ├── project/
   │ ├── data/    --> contain the dataset (downloaded from the point 3 of the setup below)
   │ ├── lib/
   │ │ ├── lib_defect_analysis.py   --> contain the functions to extract the geometric features
   │ │ └── lib_utils.py             --> contain some useful functions
   │ ├── optuna/       --> contain the scripts to run the optuna optimization
   │ ├── yolo/         --> contain the pipeline to train a yolo model from scratch
   │ ├── environment.yml
   │ └── main.ipynb    --> main jupyter notebook to train the XGBoost and LightGBM models
   ```

## Usage

### Setup
1. Clone the repository and enter the GrapheNetDefectDetector directory:
   ```bash
   git clone -b published https://github.com/daimoners/GrapheNetDefectDetector.git --depth 1 && cd GrapheNetDefectDetector
   ```

2. Create the conda env from the `environment.yaml` file and activate the conda env:
   ```bash
   conda env create -f environment.yaml && conda activate defect_analysis
   ```

3. Download the dataset and unzip it in the root folder:
   ```bash
   gdown 'https://drive.google.com/uc?id=1mDVN3YxorrBmQEofuOhyMuRVIL2AwYaa' && unzip data_chapter_7.zip
   ```

### Configuration and Train

Just run each block of the `main.ipynb` jupyter notebook (is all already configured). The notebook is explained in detail, illustrating all the various steps to train the XGBoost and LightGBM models.


### (Optional) Re-optimize the hyperparameters with Optuna

In order to optimize the XGBoost parameters for current target, the `tuner.py` script can be customized (the ranges of the hyperparameters search space) and launched from the `optuna` folder:
   
   ```bash
   cd optuna && python tuner.py
   ```

### (Optional) Re-train the YOLO model from scratch

To retrain the YOLO model, customize the `dataset.yaml` config and run the `dataset_generator.py`, in order to augment and split a dataset from a dataset exported from LabelStudio:
   ```bash
   cd yolo && python dataset_generator.py
   ```
Then, customize the `cfg.yaml` config and run the `train.py`, in order to perform the training of the YOLO model:
```bash
   python train.py
   ```
