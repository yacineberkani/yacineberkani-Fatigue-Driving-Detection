# Détection de Fatigue au Volant par Deep Learning

## Description
Système de détection de fatigue au volant basé sur l'apprentissage profond et la fusion multi-indices, utilisant MTCNN amélioré et E-MSR Net pour la détection en temps réel des signes de fatigue du conducteur.

[![Démonstration en temps réel](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=W8kyi7XJ66Y)
<video width="640" height="360" controls>
  <source src="[media/video.mp4](https://www.youtube.com/watch?v=W8kyi7XJ66Y)" type="video/mp4">
  Votre navigateur ne supporte pas la balise vidéo.
</video>


<video width="640" height="360" controls>
  <source src="https://www.youtube.com/watch?v=W8kyi7XJ66Y" type="video/mp4">
  Votre navigateur ne supporte pas la balise vidéo.
</video>


## Table des matières
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Données](#données)
4. [Utilisation](#utilisation)
5. [Dépannage](#dépannage)

## Installation

### Prérequis
- Python 3.7+
- TensorFlow 2.5+
- OpenCV 4.5+
- CUDA (recommandé)
- GPU 8GB+ VRAM (recommandé)

### Étapes d'installation
1. Clonez le dépôt :
`````
git clone https://github.com/votre-username/fatigue-detection.git
``````
`cd fatigue-detection`

2. Créez un environnement virtuel :
``````
python -m venv venv
source venv/bin/activate
``````
Windows :
``````
python -m venv venv
venv\Scripts\activate
``````
Linux/MacOS :
``````
python3 -m venv venv
source venv/bin/activate
``````

3. Installez les dépendances :
``````
pip install -r requirements.txt
``````

4. Vérifiez que les données sont présentes :
``````
python src/fatigue_detection/scripts/check_setup.py
``````  

## Structure du projet 
``````
├── data/
│ ├── custom/ # Vos données personnalisées
│ │ ├── eyes/
│ │ │ ├── open/
│ │ │ └── closed/
│ │ └── mouth/
│ │ ├── open/
│ │ └── closed/
│ ├── raw/ # Données brutes (WIDER FACE, MTFL)
│ └── processed/ # Données traitées
├── models/ # Modèles entraînés
├── logs/ # Logs d'entraînement
└── src/
└── fatigue_detection/ # Code source
``````

## Données
### 1. Préparation des données
Créer la structure des dossiers
``````
python -m src.fatigue_detection.data.create_custom_dirs
``````
Télécharger les datasets
``````
python -m src.fatigue_detection.data.download_datasets
``````

### 2. Datasets requis
- **WIDER FACE** : Pour la détection de visage
- **MTFL** : Pour les points clés du visage
- **Données personnalisées** :
  - Format : JPG
  - Taille : 224x224 pixels
  - Types : Yeux (ouverts/fermés), Bouche (ouverte/fermée)

### 3. Traitement des données
``````
python -m src.fatigue_detection.data.prepare_dataset \
--input_dir data/custom \
--output_dir data/processed \
--augment True
``````

## Utilisation

### 1. Entraînement
Entraîner MTCNN
``````
python -m src.fatigue_detection.scripts.train \
    --model_type mtcnn \
    --data_dir data/raw \
    --batch_size 32 \
    --epochs 5 \
    --output_dir models
``````

Entraîner E-MSR Net
``````
python -m src.fatigue_detection.scripts.train \
    --model_type emsr \
    --data_dir data/processed \
    --batch_size 32 \
    --epochs 50 \
    --output_dir models
``````
### 2. Évaluation
``````
python -m src.fatigue_detection.scripts.evaluate \
    --model_path models/checkpoints/best_model.keras \
    --test_dir data/processed \
    --batch_size 32 \
    --results_dir resultats
``````
### 3. Détection en temps réel

``````
python -m src.fatigue_detection.scripts.run_detection \
    --model_path models/checkpoints/best_model.keras \
    --shape_predictor shape_predictor_68_face_landmarks.dat \
    --threshold 0.25 \
    --camera 0
``````
### 3.2 Contrôles pendant l'exécution :
- 'q' : Quitter
- 'r' : Réinitialiser la détection
- Le seuil peut être ajusté avec --threshold

### 4. Détection sur des images
``````
python -m src.fatigue_detection.scripts.test_images \
    --model_path models/checkpoints/best_model.keras \
    --test_dir data/test_images \
    --output_dir resultats/tests \
    --threshold 0.25
``````
## Dépannage

### Problèmes courants
1. **Erreur GPU**
   - Vérifier l'installation CUDA
   - Vérifier la compatibilité TensorFlow/CUDA

2. **Erreur mémoire**
   - Réduire batch_size
   - Libérer mémoire GPU

3. **Erreur données**
   - Vérifier structure dossiers
   - Vérifier format images

### Support
- Ouvrir une issue sur GitHub
- Consulter la documentation
- Contacter les mainteneurs

## Licence
MIT License

## Citation

```
@article{fatigue_detection,
title={Fatigue Driving Detection Based on Deep Learning and Multi-Index Fusion},
author={HUIJIE JIA, ZHONGJUN XIAO, AND PENG JI},
journal={IEEE Access},
year={2021}
}
```
