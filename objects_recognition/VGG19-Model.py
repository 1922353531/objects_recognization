from keras.models import Model
from keras.layers import *
from keras import regularizers
from keras.utils import to_categorical
import numpy as np
import os
import cv2 as cv

input_datas = Input(shape=(settings.img_rows, settings.img_cols, settings.img_channels, ), dtype='float32')

Conv2D_1_1 = Conv2D(64, (3, 3), activation='relu')(input_datas)
Conv2D_1_2 = Conv2D(64, (3, 3), activation='relu')(Conv2D_1_1)
MaxPooling2D_1 = MaxPooling2D((7, 7), strides=(1, 1))(Conv2D_1_2)

Conv2D_2_1 = Conv2D(128, (3, 3), activation='relu')(MaxPooling2D_1)
Conv2D_2_2 = Conv2D(128, (3, 3), activation='relu')(Conv2D_2_1)
MaxPooling2D_2 = MaxPooling2D((7, 7), strides=(1, 1))(Conv2D_2_2)

Conv2D_3_1 = Conv2D(256, (3, 3), activation='relu')(MaxPooling2D_2)
Conv2D_3_2 = Conv2D(256, (3, 3), activation='relu')(Conv2D_3_1)
Conv2D_3_3 = Conv2D(256, (3, 3), activation='relu')(Conv2D_3_2)
Conv2D_3_4 = Conv2D(256, (3, 3), activation='relu')(Conv2D_3_3)
MaxPooling2D_3 = MaxPooling2D((7, 7), strides=(1, 1))(Conv2D_3_4)

Conv2D_4_1 = Conv2D(512, (3, 3), activation='relu')(MaxPooling2D_3)
Conv2D_4_2 = Conv2D(512, (3, 3), activation='relu')(Conv2D_4_1)
Conv2D_4_3 = Conv2D(512, (3, 3), activation='relu')(Conv2D_4_2)
Conv2D_4_4 = Conv2D(512, (3, 3), activation='relu')(Conv2D_4_3)
MaxPooling2D_4 = MaxPooling2D((7, 7), strides=(1, 1))(Conv2D_4_4)
Conv2D_4_5 = Conv2D(512, (3, 3), activation='relu')(MaxPooling2D_4)
Conv2D_4_6 = Conv2D(512, (3, 3), activation='relu')(Conv2D_4_5)
Conv2D_4_7 = Conv2D(512, (3, 3), activation='relu')(Conv2D_4_6)
Conv2D_4_8 = Conv2D(512, (3, 3), activation='relu')(Conv2D_4_7)
MaxPooling2D_5 = MaxPooling2D((7, 7), strides=(1, 1))(Conv2D_4_8)

Flatten_1 = Flatten()(MaxPooling2D_5)
Dense_1 = Dense(4096, activation='relu', activity_regularizer=regularizers.l1(settings.re_lamda))(Flatten_1)
Dropout_1 = Dropout(settings.dropout_rate)(Dense_1)
Dense_2 = Dense(4096, activation='relu', activity_regularizer=regularizers.l1(settings.re_lamda))(Dropout_1)
Dropout_2 = Dropout(settings.dropout_rate)(Dense_2)
Dense_3 = Dense(1000, activation='softmax')(Dropout_2)
output_datas = Dense_3

vgg19_model = Model(input_datas, output_datas)
