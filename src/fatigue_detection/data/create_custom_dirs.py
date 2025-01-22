from pathlib import Path

def create_custom_directories(base_dir):
    """Crée la structure de dossiers pour les données personnalisées"""
    base_dir = Path(base_dir)
    custom_dir = base_dir / 'custom'
    
    # Structure des dossiers
    directories = [
        custom_dir / 'eyes' / 'open',
        custom_dir / 'eyes' / 'closed',
        custom_dir / 'mouth' / 'open',
        custom_dir / 'mouth' / 'closed'
    ]
    
    # Création des dossiers
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Créé: {directory}")
        
    return custom_dir

if __name__ == '__main__':
    # Chemin vers le dossier data de votre projet
    data_dir = Path(__file__).parent.parent.parent.parent / 'data'
    custom_dir = create_custom_directories(data_dir)
    
    print("\nStructure créée avec succès!")
    print("Placez vos images dans les dossiers correspondants:")
    print(f"- Yeux ouverts: {custom_dir}/eyes/open/")
    print(f"- Yeux fermés: {custom_dir}/eyes/closed/")
    print(f"- Bouche ouverte: {custom_dir}/mouth/open/")
    print(f"- Bouche fermée: {custom_dir}/mouth/closed/") 