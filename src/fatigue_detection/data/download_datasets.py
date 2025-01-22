import wget
import os
import zipfile
from pathlib import Path
import gdown  # Pour télécharger depuis Google Drive

def download_wider_face(output_dir):
    """Télécharge et extrait WIDER FACE dataset"""
    output_dir = Path(output_dir)
    wider_face_dir = output_dir / 'raw' / 'wider_face'
    wider_face_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs des fichiers (Google Drive)
    files = {
        'WIDER_train.zip': 'https://drive.google.com/uc?id=0B6eKvaijfFUDQUUwd21EckhUbWs',
        'WIDER_val.zip': 'https://drive.google.com/uc?id=0B6eKvaijfFUDd3dIRmpvSk8tLUk',
        'wider_face_split.zip': 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
    }
    
    for filename, url in files.items():
        output_path = wider_face_dir / filename
        if not output_path.exists():
            print(f"\nTéléchargement de {filename}...")
            try:
                if 'drive.google.com' in url:
                    gdown.download(url, str(output_path), quiet=False)
                else:
                    wget.download(url, str(output_path))
                
                print(f"\nExtraction de {filename}...")
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(str(wider_face_dir))
            except Exception as e:
                print(f"Erreur lors du téléchargement/extraction de {filename}: {e}")
                continue
    
    print("\nVérification de la structure des données...")
    required_dirs = [
        wider_face_dir / 'WIDER_train' / 'images',
        wider_face_dir / 'WIDER_val' / 'images',
        wider_face_dir / 'wider_face_split'
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            print(f"ATTENTION: Dossier manquant: {directory}")
            return False
    
    return True

def download_mtfl(output_dir):
    """Télécharge et extrait MTFL dataset"""
    output_dir = Path(output_dir)
    mtfl_dir = output_dir / 'raw' / 'mtfl'
    mtfl_dir.mkdir(parents=True, exist_ok=True)
    
    # URL mise à jour pour MTFL
    url = "http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip"
    
    # Téléchargement
    zip_path = mtfl_dir / "MTFL.zip"
    if not zip_path.exists():
        print("Téléchargement de MTFL...")
        try:
            wget.download(url, str(zip_path))
            print("\nExtraction de MTFL...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(str(mtfl_dir))
        except Exception as e:
            print(f"Erreur lors du téléchargement de MTFL: {e}")
            print("Vous pouvez télécharger manuellement depuis: http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html")

def download_alternative():
    """Message pour le téléchargement manuel"""
    print("""
    Les datasets peuvent être téléchargés manuellement depuis:
    
    1. WIDER FACE:
       - Site officiel: http://shuoyang1213.me/WIDERFACE/
       - Téléchargez:
         * WIDER_train.zip
         * WIDER_val.zip
         * WIDER_test.zip
         * wider_face_split.zip
    
    2. MTFL:
       - Site officiel: http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
       - Téléchargez MTFL.zip
    
    Placez les fichiers téléchargés dans:
    - data/raw/wider_face/
    - data/raw/mtfl/
    """)

if __name__ == "__main__":
    # Si le téléchargement automatique échoue, afficher les instructions manuelles
    download_alternative() 