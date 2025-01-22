import tensorflow as tf

class EMSRNet(tf.keras.Model):
    def __init__(self, num_classes=2, **kwargs):
        super(EMSRNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        # Première couche conv
        self.conv1 = tf.keras.layers.Conv2D(32, 3, 2, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        
        # Couches convolutionnelles
        self.conv_layers = [
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25)
        ]
        
        # Couches denses
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        # Premier bloc
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        # Blocs convolutionnels
        for layer in self.conv_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Couches denses
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        
        return x
    
    def get_config(self):
        config = super(EMSRNet, self).get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def model(self):
        """Retourne un modèle fonctionnel équivalent"""
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)) 