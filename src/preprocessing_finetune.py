import tensorflow as tf
from tensorflow.keras import Sequential, layers

def get_train_datagen(config):
    aug = config['augmentation']
    return Sequential([
        # 1) Normalisation
        layers.Rescaling(1./255),

        # 2) Géométrie
        layers.RandomRotation(aug['rotation_range'] / 360.0),
        layers.RandomTranslation(aug['height_shift_range'], aug['width_shift_range']),
        layers.RandomZoom(aug['zoom_range']),
        layers.RandomContrast(aug['contrast_range'][1] - aug['contrast_range'][0], # écart max
                               lower=aug['contrast_range'][0]),

        # 3) Couleurs
        layers.RandomBrightness(factor=aug['brightness_range'][1] - aug['brightness_range'][0],
                                value_range=aug['brightness_range']),

        # 4) Flip
        layers.RandomFlip("horizontal") if aug['horizontal_flip'] else layers.Layer(),

        # 5) Cutout
        layers.Lambda(
            lambda x: tf.image.random_cutout(
                x,
                mask_size=aug['cutout_size'],
                constant_values=0.0
            )
        ),
    ])

def get_val_datagen():
    return Sequential([
        layers.Rescaling(1./255)
    ])
