## GrapheNetDefectDetector
Ho fatto il tuning dei parametri solo per il target total_energy, bisognerebbe farlo anche per energy_per_atomo o qualche altro target che vogliamo studiare.

attualmente il file ``tuner.py``, ha come modello al suo interno solo il xgbBoost  e salva tutto in ``optuna.log``. Per usare il tuner ho salvato i dati di training normalizzati con numpy e si chiamano
``X_norm.npy`` e ``Y_norm.npy``.

Si potrebbe eliminare tutta la parte sotto il fit che è la roba vecchia del tesista.


