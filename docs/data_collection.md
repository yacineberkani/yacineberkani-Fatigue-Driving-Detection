# Sources de données

1. WIDER FACE Dataset
- URL: http://shuoyang1213.me/WIDERFACE/
- Utilisation: Entraînement de MTCNN pour la détection de visage
- Format: Images .jpg avec annotations

2. MTFL Dataset (Multi-Task Facial Landmark)
- URL: http://mmlab.ie.cuhk.edu.hk/projects/MTFL.html
- Utilisation: Entraînement des points clés du visage
- Format: Images .jpg avec fichiers d'annotation

3. Pour les données personnalisées (yeux/bouche):
- Sources possibles:
  * MRL Eye Dataset
  * CEW Dataset (Closed Eyes in the Wild)
  * YAWDD Dataset (Yawning Detection Dataset)
  
# Préparation des données

1. Images des yeux:
- Taille: 224x224 pixels
- Classes: ouvert/fermé
- Augmentation recommandée:
  * Rotation ±15°
  * Variation de luminosité
  * Flou gaussien léger

2. Images de la bouche:
- Taille: 224x224 pixels
- Classes: ouvert (bâillement)/fermé
- Augmentation recommandée:
  * Rotation ±10°
  * Variation de contraste
  * Ajout de bruit gaussien 