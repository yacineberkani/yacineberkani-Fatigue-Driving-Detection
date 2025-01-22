from pathlib import Path
import sys
import os

# Ajouter le chemin du projet à PYTHONPATH
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

from src.fatigue_detection.data.download_datasets import download_wider_face, download_mtfl

def main():
    # Chemin du dossier data
    data_dir = Path(project_root) / 'data'
    
    # Téléchargement des datasets
    download_wider_face(data_dir)
    download_mtfl(data_dir)
    
    print("\nTéléchargement terminé!")
    print(f"Les données sont dans: {data_dir}")

if __name__ == '__main__':
    main() 