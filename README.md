# Data-Driven Analysis and Generation of Defective Graphene Nanoflakes for Property Prediction and Device Optimization
<!-- Ho fatto il tuning dei parametri solo per il target total_energy, bisognerebbe farlo anche per energy_per_atom o qualche altro target che vogliamo studiare.

attualmente il file ``tuner.py``, ha come modello al suo interno solo il xgBoost  e salva tutto in ``optuna.log``. Per usare il tuner ho salvato i dati di training normalizzati con numpy e si chiamano
``X_norm.npy`` e ``Y_norm.npy``.

Si potrebbe eliminare tutta la parte sotto il fit che è la roba vecchia del tesista. -->
 
This repo is about using computer vision techniques (and in particular convolutional neural networks) in order to extract geometrical and frequency features from images and use such features with classical ML techniques to predict the electronic properties of defected graphene flakes. The framework take care to create the dataset starting from a folder of **.xyz** files and a **.csv** file containing the target properties, and uses YOLO for defect detection and XGBoost for target prediction.

## Project Structure
   ```.
   ├── project/
   │ ├── cfg/
   │ │ └── config.yaml
   │ ├── dataset/
   │ │ ├── signals_train.pickle
   │ │ ├── signals_val.pickle
   │ │ └── signals_test.pickle
   │ ├── lib/
   │ │ └── utils.py
   │ ├── logs/ (folder where the logs will be saved during training/inference)
   │ ├── models/ (folder where the models will be saved during training)
   │ ├── results/
   │ │ ├── baseline/ (contain the results of the baseline model)
   │ │ └── optimized/ (contain the results of the optimized model)
   │ ├── outputs/ (folder created by hydra containing the experiments hystory)
   │ ├── conda_environment.yml
   │ ├── requirements.txt
   │ ├── signals.pickle
   │ ├── question_1.py
   │ └── question_2.py
   ```
- The core functionality of this project is provided by the `question_1.py` and `question_2.py` scripts, which will be discussed below. The `utils.py` script, located in the `lib` folder, contains essential training and evaluation utilities, including model checkpointing, performance logging, and standard PyTorch training algorithms. To keep the main scripts focused on the assignment, all non-assignment-specific code has been delegated to this library. Also, `utils.py` includes the implementation of a metrics tracking class for monitoring performance during training and evaluation, as well as an early stopping class.

- The `question_2.py` script, in particular, contains the `Code128Loss`, `Code128Model`, and `SignalDataset` classes discussed in the PDF of the solutions.

- The `dataset` folder contains the split dataset used for training and evaluating the proposed model. You could also create your own split, by running the `question_1.py` script.

- The `results` folder contain the trained models and performance metrics (inside the log files) of the baseline and the optimized models.

- The `models` folder is the default directory where the models will be saved during the training procedure.

## Usage

### Setup
1. Clone the repository and enter the GrapheNetDefectDetector directory:

   ```bash
   git clone -b published https://github.com/daimoners/GrapheNetDefectDetector.git --depth 1 && cd GrapheNetDefectDetector
   ```

2. Create the conda env from the environment.yaml file:

   ```bash
   conda env create -f environment.yaml
   ```

3. Activate the conda env:

   ```bash
   conda activate yolo
   ```

### Configuration and Train

1. Customize the paths in the first block of **main.ipynb** according to your needs

2. Run the following blocks to train and evaluate the model

### (Optional) Train the YOLO model from scratch

In order to optimize the XGBoost parameters for each target, the **tuner.py** script can be launched from the **optuna** folder:
   
   ```bash
   cd optuna && python tuner.py
   ```

### (Optional) Optimization

In order to optimize the XGBoost parameters for each target, the **tuner.py** script can be launched from the **optuna** folder:
   
   ```bash
   cd optuna && python tuner.py
   ```