# -*- coding: utf-8 -*-
import os
import warnings
from tensorflow.keras import models
from tensorflow.keras import backend
from tensorflow.keras import layers
import tensorflow.keras.utils as keras_utils

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters1, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters1, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    shortcut = layers.Conv2D(filters2, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    return x


def ResNet18v2(input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=3,
             **kwargs):

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 256], stage=2, block='b')

    x = conv_block(x, 3, [128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 512], stage=3, block='b')

    x = conv_block(x, 3, [256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 1024], stage=4, block='b')

    x = conv_block(x, 3, [512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 2048], stage=5, block='b')

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet18')

    return model
