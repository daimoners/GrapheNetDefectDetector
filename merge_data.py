from pathlib import Path
import shutil
from tqdm.rich import tqdm

# Funzione per unire le cartelle
def merge_folders(source_dirs, dest_dir, subdirs):
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    
    for subdir in subdirs:
        dest_subdir = dest_dir / subdir
        if not dest_subdir.exists():
            dest_subdir.mkdir(parents=True)
        
        for source_dir in source_dirs:
            source_subdir = source_dir / subdir
            if source_subdir.exists():
                items = list(source_subdir.iterdir())
                for item in tqdm(items, desc=f"Merging {subdir} from {source_dir.name}", leave=False):
                    dest_item = dest_subdir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest_item)

# Directory corrente
current_dir = Path.cwd()

# Trova tutte le cartelle data_batch_xxx, escludendo data_batch_full
data_batches = sorted(dir for dir in current_dir.glob("data_batch_*") if dir.is_dir() and dir.name != "data_batch_full" and dir.name != "data_batch_1")

# Cartella di destinazione
dest_folder = current_dir / "data_234"

# Sottocartelle da unire
subfolders_to_merge = ["xyz_files", "transport_output", "json_files"]

# Unisci le cartelle
merge_folders(data_batches, dest_folder, subfolders_to_merge)

print("Unione completata!")
