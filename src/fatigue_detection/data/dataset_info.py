DATASET_INFO = {
    # Pour l'entraînement MTCNN
    'wider_face': {
        'description': 'Dataset pour la détection de visage',
        'url': 'http://shuoyang1213.me/WIDERFACE/',
        'nb_images': 32203,
        'format': '.jpg'
    },
    
    'mtfl': {
        'description': 'Dataset pour les points clés du visage',
        'url': 'https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html',
        'nb_images': 12995,
        'format': '.jpg',
        'annotations': '.txt'  # Format: x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
    },
    
    # Pour l'entraînement E-MSR Net
    'custom_dataset': {
        'eyes': {
            'open': {
                'nb_required': 1500,
                'format': '.jpg',
                'size': '224x224'
            },
            'closed': {
                'nb_required': 1374,
                'format': '.jpg',
                'size': '224x224'
            }
        },
        'mouth': {
            'open': {
                'nb_required': 9246,
                'format': '.jpg',
                'size': '224x224'
            },
            'closed': {
                'nb_required': 9701,
                'format': '.jpg',
                'size': '224x224'
            }
        }
    }
} 