from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_datagen(config):
    aug = config['augmentation']
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=aug['rotation_range'],
        width_shift_range=aug['width_shift_range'],
        height_shift_range=aug['height_shift_range'],
        shear_range=aug['shear_range'],
        zoom_range=aug['zoom_range'],
        horizontal_flip=aug['horizontal_flip'],
        brightness_range=aug['brightness_range']
    )

def get_val_datagen():
    return ImageDataGenerator(
        rescale=1./255
    )
