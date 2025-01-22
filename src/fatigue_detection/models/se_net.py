import tensorflow as tf

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Squeeze
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # Excitation
        self.fc1 = tf.keras.layers.Dense(channels // reduction_ratio)
        self.relu = tf.keras.layers.ReLU()
        self.fc2 = tf.keras.layers.Dense(channels)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Squeeze
        x = self.global_pool(inputs)
        
        # Excitation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        # Reshape pour la multiplication
        x = tf.reshape(x, [batch_size, 1, 1, self.channels])
        
        return inputs * x 