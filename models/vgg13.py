from loaders.records import TfRecordData
from paths import *
import tensorflow as tf
from loaders.dataloader import *
import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

def vgg13():
    alpha = 0.00002

    # layer1 32*32*3
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                     input_shape=(252, 448, 3), kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    # layer2 32*32*64
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # layer3 16*16*64
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # layer4 16*16*128
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # layer5 8*8*128
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # layer6 8*8*256
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    # layer7 8*8*256
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # layer8 4*4*256
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # layer9 4*4*512
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    # layer10 4*4*512
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # layer11 2*2*512
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # layer12 2*2*512
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(alpha)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    # layer13 2*2*512
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    # layer14 1*1*512
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    # layer15 512
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    # layer16 512
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(7))
    model.add(tf.keras.layers.Activation('softmax'))
    # 10
    model.compile(loss='cosine_similarity', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    return model



def main():
    desc_fp = Datasets.phase1['descriptors']
    video_fp = Datasets.phase1['videos']
    motion_fp = Datasets.phase1['motion_capture']

    tfd = TfRecordData(desc_fp, video_fp, motion_fp)

    train_fp = Datasets.phase1['train_records']
    validation_fp = Datasets.phase1['validation_records']

    raw_train_dataset = tf.data.TFRecordDataset(train_fp)
    raw_validation_dataset = tf.data.TFRecordDataset(validation_fp)

    parsed_train_dataset = raw_train_dataset.map(tfd.decode)
    """
    for img, label in parsed_train_dataset.take(1):
        img = img.numpy()[0]
        display_frame(img)
        label = 2
    """
    parsed_validation_dataset = raw_validation_dataset.map(tfd.decode)

    x_train = [] #images
    y_train = [] #labels
    image_shape = None
    for image, label in parsed_validation_dataset.take(40):
        list_frames = image.numpy()
        if image_shape is None:
            image_shape = list_frames[0].shape
        for i in range(len(list_frames)):
            x_train.append(list_frames[i])
            y_train.append(label.numpy())

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #(252,448,3)

    train_dataset = parsed_train_dataset.shuffle(buffer_size=64).batch(64)

    #test_dataset = parsed_image_dataset.take(1000).batch(64)
    #train_dataset = parsed_image_dataset.skip(1000).shuffle(buffer_size=64).batch(64)

    model = vgg13()
    model.summary()

    model.fit(x_train, y_train, epochs=1)



if __name__ == '__main__':
    main()
