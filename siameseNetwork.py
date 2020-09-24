import os
import numpy as np
import tensorflow as tf


K = tf.keras.backend

def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp

class SiameseNetwork(tf.keras.Model):
    def __init__(self, vgg_face):
        super(SiameseNetwork, self).__init__()
        self.vgg_face = vgg_face
        
    @tf.function
    def call(self, inputs):
        image_1, image_2, image_3 =  inputs
        with tf.name_scope("Anchor") as scope:
            feature_1 = self.vgg_face(image_1)
            feature_1 = tf.math.l2_normalize(feature_1, axis=-1)
        with tf.name_scope("Positive") as scope:
            feature_2 = self.vgg_face(image_2)
            feature_2 = tf.math.l2_normalize(feature_2, axis=-1)
        with tf.name_scope("Negative") as scope:
            feature_3 = self.vgg_face(image_3)
            feature_3 = tf.math.l2_normalize(feature_3, axis=-1)
        return [feature_1, feature_2, feature_3]
    
    @tf.function
    def get_features(self, inputs):
        return tf.math.l2_normalize(self.vgg_face(inputs), axis=-1)

def get_siamese_model():
    vggface = tf.keras.models.Sequential()
    vggface.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME", input_shape=(224,224, 3)))
    vggface.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    
    vggface.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    
    vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
    
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
    vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    vggface.add(tf.keras.layers.Flatten())

    vggface.add(tf.keras.layers.Dense(4096, activation='relu'))
    vggface.add(tf.keras.layers.Dropout(0.5))
    vggface.add(tf.keras.layers.Dense(4096, activation='relu'))
    vggface.add(tf.keras.layers.Dropout(0.5))
    vggface.add(tf.keras.layers.Dense(2622, activation='softmax'))

    vggface.pop()
    vggface.add(tf.keras.layers.Dense(128, use_bias=False))

    for layer in vggface.layers[:-2]:
        layer.trainable = False

    model = SiameseNetwork(vggface)

    base_dir = "."
    checkpoint_path = os.path.join(base_dir, 'logs/model/siamese-1')

    _ = model([tf.zeros((1,224,224,3)), tf.zeros((1,224,224,3)), tf.zeros((1,224,224,3))])
    _ = model.get_features(tf.zeros((1,224,224,3)))

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path)

    return model