from keras.layers import *
from keras.utils import to_categorical
from keras.models import Model
import numpy as np
import keras.backend as K
import os
import cv2 as cv

RESNET_V2_A_COUNT = 0
RESNET_V2_B_COUNT = 0
RESNET_V2_C_COUNT = 0

def resnet_v2_stem(x_input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    with K.name_scope("stem"):
        x = Conv2D(32, (3, 3), activation="relu", strides=(2, 2))(x_input)  # 149 * 149 * 32
        x = Conv2D(32, (3, 3), activation="relu")(x)  # 147 * 147 * 32
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)  # 147 * 147 * 64

        x1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x2 = Conv2D(96, (3, 3), activation="relu", strides=(2, 2))(x)

        x = concatenate([x1, x2], axis=-1)  # 73 * 73 * 160

        x1 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
        x1 = Conv2D(96, (3, 3), activation="relu")(x1)

        x2 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
        x2 = Conv2D(64, (7, 1), activation="relu", padding="same")(x2)
        x2 = Conv2D(64, (1, 7), activation="relu", padding="same")(x2)
        x2 = Conv2D(96, (3, 3), activation="relu", padding="valid")(x2)

        x = concatenate([x1, x2], axis=-1)  # 71 * 71 * 192

        x1 = Conv2D(192, (3, 3), activation="relu", strides=(2, 2))(x)

        x2 = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = concatenate([x1, x2], axis=-1)  # 35 * 35 * 384

        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x


def inception_resnet_v2_A(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.'''
    global RESNET_V2_A_COUNT
    RESNET_V2_A_COUNT += 1
    with K.name_scope('inception_resnet_v2_A' + str(RESNET_V2_A_COUNT)):
        ar1 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)

        ar2 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
        ar2 = Conv2D(32, (3, 3), activation="relu", padding="same")(ar2)

        ar3 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
        ar3 = Conv2D(48, (3, 3), activation="relu", padding="same")(ar3)
        ar3 = Conv2D(64, (3, 3), activation="relu", padding="same")(ar3)

        merged = concatenate([ar1, ar2, ar3], axis=-1)

        ar = Conv2D(384, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual:
            ar = Lambda(lambda a: a * 0.1)(ar)

        x = add([x_input, ar])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x


def inception_resnet_v2_B(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.'''
    global RESNET_V2_B_COUNT
    RESNET_V2_B_COUNT += 1
    with K.name_scope('inception_resnet_v2_B' + str(RESNET_V2_B_COUNT)):
        br1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

        br2 = Conv2D(128, (1, 1), activation="relu", padding="same")(x_input)
        br2 = Conv2D(160, (1, 7), activation="relu", padding="same")(br2)
        br2 = Conv2D(192, (7, 1), activation="relu", padding="same")(br2)

        merged = concatenate([br1, br2], axis=-1)

        br = Conv2D(1152, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual:
            br = Lambda(lambda b: b * 0.1)(br)

        x = add([x_input, br])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x


def inception_resnet_v2_C(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.'''
    global RESNET_V2_C_COUNT
    RESNET_V2_C_COUNT += 1
    with K.name_scope('inception_resnet_v2_C' + str(RESNET_V2_C_COUNT)):
        cr1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

        cr2 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
        cr2 = Conv2D(224, (1, 3), activation="relu", padding="same")(cr2)
        cr2 = Conv2D(256, (3, 1), activation="relu", padding="same")(cr2)

        merged = concatenate([cr1, cr2], axis=-1)

        cr = Conv2D(2144, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual:
            cr = Lambda(lambda c: c * 0.1)(cr)

        x = add([x_input, cr])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x

def reduction_resnet_v2_A(x_input):
    with K.name_scope('reduction_resnet_A'):
        ra1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

        ra2 = Conv2D(384, (3, 3), activation='relu', strides=(2, 2), padding='valid')(x_input)

        ra3 = Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
        ra3 = Conv2D(256, (3, 3), activation='relu', padding='same')(ra3)
        ra3 = Conv2D(384, (3, 3), activation='relu', strides=(2, 2), padding='valid')(ra3)

        merged_vector = concatenate([ra1, ra2, ra3], axis=-1)

        x = BatchNormalization(axis=-1)(merged_vector)
        x = Activation('relu')(x)
    return x

def reduction_resnet_v2_B(x_input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.'''
    with K.name_scope('reduction_resnet_v2_B'):
        rbr1 = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x_input)

        rbr2 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr2 = Conv2D(384, (3, 3), activation="relu", strides=(2, 2))(rbr2)

        rbr3 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr3 = Conv2D(288, (3, 3), activation="relu", strides=(2, 2))(rbr3)

        rbr4 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr4 = Conv2D(288, (3, 3), activation="relu", padding="same")(rbr4)
        rbr4 = Conv2D(320, (3, 3), activation="relu", strides=(2, 2))(rbr4)

        merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis=-1)
        rbr = BatchNormalization(axis=-1)(merged)
        rbr = Activation("relu")(rbr)
    return rbr


def inception_resnet_v2(scale=True):
    '''Creates the Inception_ResNet_v1 network.'''

    init = Input((settings.img_rows, settings.img_cols, 3, ))  # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering

    # Input shape is 299 * 299 * 3
    x = resnet_v2_stem(init)  # Output: 35 * 35 * 256

    # 5 x Inception A
    for i in range(5):
        x = inception_resnet_v2_A(x, scale_residual=scale)
        # Output: 35 * 35 * 256

    # Reduction A
    x = reduction_resnet_v2_A(x)  # Output: 17 * 17 * 896

    # 10 x Inception B
    for i in range(10):
        x = inception_resnet_v2_B(x, scale_residual=scale)
        # Output: 17 * 17 * 896

    # Reduction B
    x = reduction_resnet_v2_B(x)  # Output: 8 * 8 * 1792

    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v2_C(x, scale_residual=scale)
        # Output: 8 * 8 * 1792

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)  # Output: 1792

    # Dropout
    x = Dropout(0.2)(x)  # Keep dropout 0.2 as mentioned in the paper
    x = Flatten()(x)  # Output: 1792

    # Output layer
    output = Dense(units=len(data_setting.objects_num), activation="softmax")(x)  # Output: 10000

    model = Model(init, output, name="Inception-ResNet-v2")

    return model


inception_resnet_v2_model = inception_resnet_v2()


