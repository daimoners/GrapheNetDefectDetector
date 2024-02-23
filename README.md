## GrapheNetDefectDetector
<!-- Ho fatto il tuning dei parametri solo per il target total_energy, bisognerebbe farlo anche per energy_per_atom o qualche altro target che vogliamo studiare.

attualmente il file ``tuner.py``, ha come modello al suo interno solo il xgBoost  e salva tutto in ``optuna.log``. Per usare il tuner ho salvato i dati di training normalizzati con numpy e si chiamano
``X_norm.npy`` e ``Y_norm.npy``.

Si potrebbe eliminare tutta la parte sotto il fit che è la roba vecchia del tesista. -->
 
This repo is about using computer vision techniques (and in particular convolutional neural networks) in order to extract geometrical features from images and use such features with classical ML techniques to predict the electronic properties of defected graphene flakes. The framewrok take care to create the dataset starting from a folder of **.xyz** files and a **.csv** file containing the target properties, and uses YOLO for defect detection and XGBoost for target prediction.

## Requirements

* CUDA
* miniconda3

## Usage

### Setup
1. Clone the repository and enter the GrapheNetDefectDetector directory:

   ```bash
   git clone -b published https://github.com/daimoners/GrapheNetDefectDetector.git && cd GrapheNetDefectDetector
   ```

2. Create the conda env from the conda_env.yaml file:

   ```bash
   conda env create -f conda_env.yaml
   ```

3. Activate the conda env:

   ```bash
   conda activate yolo
   ```

### Configuration and Train

1. Customize the paths in the first block of **main.ipynb** according to your needs

2. Run the following blocks to train and evaluate the model


### Optimization

In order to optimize the XGBoost parameters for each target, the **tuner.py** script can be launched from the **optuna** folder:
   
   ```bash
   cd optuna && python tuner.py
   ```

### Test with Simulated STM Images

TBD.


