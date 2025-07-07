import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential, layers
from PIL import Image
import io

def _decode_tiff_and_resize(image_bytes, img_size):
    """
    Décode un flux bytes d’une image TIFF, convertit en RGB,
    redimensionne à img_size (h, w) et renvoie un numpy array uint8.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    # PIL resize prend (width, height)
    img = img.resize((img_size[1], img_size[0]), resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def cutout(image, mask_size):
    """
    Applique un Cutout de taille mask_size=(h, w) pixels sur `image` (float [0–1]).
    """
    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]
    mask_h = mask_size[0]
    mask_w = mask_size[1]

    # centre aléatoire du cutout
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)

    # bornes du cutout (clamp)
    y1 = tf.clip_by_value(cy - mask_h // 2, 0, img_h)
    x1 = tf.clip_by_value(cx - mask_w // 2, 0, img_w)
    y2 = tf.clip_by_value(y1 + mask_h, 0, img_h)
    x2 = tf.clip_by_value(x1 + mask_w, 0, img_w)

    # masque plein de 1 dans la zone, 0 ailleurs
    mask = tf.ones((y2 - y1, x2 - x1), dtype=image.dtype)
    mask = tf.pad(mask,
                  [[y1, img_h - y2],
                   [x1, img_w - x2]])
    mask = tf.expand_dims(mask, axis=-1)  # shape = (H, W, 1)

    # on noircit la zone (1-mask) * image
    return image * (1.0 - mask)


def get_train_datagen(config):
    aug = config['augmentation']

    contrast_delta = max(1 - aug['contrast_range'][0], aug['contrast_range'][1] - 1)
    brightness_delta = max(1 - aug['brightness_range'][0], aug['brightness_range'][1] - 1)

    return Sequential([
        # 1) Normalisation
        layers.Rescaling(1./255),

        # 2) Géométrie
        layers.RandomRotation(aug['rotation_range'] / 360.0),
        layers.RandomTranslation(aug['height_shift_range'], aug['width_shift_range']),
        layers.RandomZoom(aug['zoom_range']),

        # 3) Couleurs
        layers.RandomContrast(contrast_delta),
        layers.RandomBrightness(brightness_delta),

        # 4) Flip
        layers.RandomFlip("horizontal") if aug['horizontal_flip'] else layers.Layer(),

        # 5) Cutout
        layers.Lambda(
            lambda x: cutout(x, tuple(aug['cutout_size']))
        )

    ])

def get_val_datagen():
    return Sequential([
        layers.Rescaling(1./255)
    ])

def make_dataset(
    df,
    img_size=(224, 224),
    batch_size=32,
    shuffle=True,
    augment_fn=None,
    weights=None
):
    paths = df['filename'].values
    labels = pd.Categorical(df['class']).codes

    if weights is not None:
        sample_weights = df[weights].values
    else:
        sample_weights = None

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, sample_weights if sample_weights is not None else tf.zeros_like(labels)))

    @tf.autograph.experimental.do_not_convert
    def _load_img(path, label, weight, img_size=img_size):
        img_bytes = tf.io.read_file(path)
        img = tf.numpy_function(
            _decode_tiff_and_resize,
        [img_bytes, img_size],
            tf.uint8
        )
        img.set_shape([img_size[0], img_size[1], 3])
        return img, label, weight


    ds = ds.map(_load_img, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_fn is not None:
        ds = ds.map(lambda x, y, w: (augment_fn(x, training=True), y, w), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

