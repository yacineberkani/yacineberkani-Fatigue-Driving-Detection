import tensorflow as tf

class ImprovedMTCNN(tf.keras.Model):
    def __init__(self):
        super(ImprovedMTCNN, self).__init__()
        self.pnet = self._build_pnet()
        self.rnet = self._build_rnet()
        self.onet = self._build_onet()
        
    def _build_pnet(self):
        """Construit le P-Net"""
        inputs = tf.keras.layers.Input(shape=(None, None, 3))
        x = tf.keras.layers.Conv2D(10, 3, padding='valid')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(16, 3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        
        x = tf.keras.layers.Conv2D(32, 3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        
        # Ajout d'un Global Average Pooling pour obtenir une sortie 2D
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        classifier = tf.keras.layers.Dense(2, activation='softmax', name='cls')(x)
        bbox_regressor = tf.keras.layers.Dense(4, name='bbox')(x)
        
        return tf.keras.Model(inputs, [classifier, bbox_regressor], name='pnet')
        
    def _build_rnet(self):
        """Construit le R-Net"""
        inputs = tf.keras.layers.Input(shape=(24, 24, 3))
        x = tf.keras.layers.Conv2D(28, 3, padding='valid')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(48, 3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(64, 2, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.PReLU()(x)
        
        classifier = tf.keras.layers.Dense(2, activation='softmax', name='cls')(x)
        bbox_regressor = tf.keras.layers.Dense(4, name='bbox')(x)
        
        return tf.keras.Model(inputs, [classifier, bbox_regressor], name='rnet')
        
    def _build_onet(self):
        """Construit le O-Net"""
        inputs = tf.keras.layers.Input(shape=(48, 48, 3))
        x = tf.keras.layers.Conv2D(32, 3, padding='valid')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(64, 3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(64, 3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(128, 2, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.PReLU()(x)
        
        classifier = tf.keras.layers.Dense(2, activation='softmax', name='cls')(x)
        bbox_regressor = tf.keras.layers.Dense(4, name='bbox')(x)
        landmarks = tf.keras.layers.Dense(10, name='landmarks')(x)
        
        return tf.keras.Model(inputs, [classifier, bbox_regressor, landmarks], name='onet')
    
    def _resize_for_rnet(self, images):
        """Redimensionne les images pour le R-Net"""
        return tf.image.resize(images, (24, 24))
    
    def _resize_for_onet(self, images):
        """Redimensionne les images pour le O-Net"""
        return tf.image.resize(images, (48, 48))
        
    def call(self, inputs, training=None):
        """Forward pass"""
        # P-Net (maintenant avec sortie 2D grâce au GlobalAveragePooling)
        p_cls, _ = self.pnet(inputs, training=training)
        
        # R-Net
        r_input = self._resize_for_rnet(inputs)
        r_cls, _ = self.rnet(r_input, training=training)
        
        # O-Net
        o_input = self._resize_for_onet(inputs)
        o_cls, _, _ = self.onet(o_input, training=training)
        
        # Moyenne des classifications (maintenant toutes en 2D)
        final_cls = (p_cls + r_cls + o_cls) / 3.0
        return final_cls 