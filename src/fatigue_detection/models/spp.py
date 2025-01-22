class SpatialPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, pool_list=[1, 2, 4]):
        super().__init__()
        self.pool_list = pool_list

    def call(self, inputs):
        outputs = []
        for pool_size in self.pool_list:
            x = tf.keras.layers.MaxPooling2D(
                pool_size=(pool_size, pool_size),
                strides=(pool_size, pool_size)
            )(inputs)
            outputs.append(x)
        return tf.concat(outputs, axis=1) 