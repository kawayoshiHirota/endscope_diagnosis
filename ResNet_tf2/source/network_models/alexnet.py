import os
import warnings
from tensorflow.keras import models
from tensorflow.keras import backend
from tensorflow.keras import layers
import tensorflow.keras.utils as keras_utils

"""
from tensorflow.keras.initializers import TruncatedNormal, Constant
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
#from keras.callbacks import Callback, EarlyStopping
#from keras.utils.np_utils import to_categorical

def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )
def AlexNet(input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3):
    model = Sequential()

    # 第1畳み込み層
    #model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=input_shape))
    model.add(conv2d(64, 11, strides=(4,4), bias_init=0, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(192, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第３~5畳み込み層
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層
    model.add(Dense(3, activation='softmax'))

    #model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
"""
def AlexNet(input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3):
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
    

    x = layers.Conv2D(96, (11, 11),strides=(4, 4),
                      padding='valid',activation='relu',name='conv1')(img_input)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='valid',name='pool1')(x)
    x = layers.Conv2D(256, (5, 5),activation='relu',padding='same',name='conv2')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='valid',name='pool2')(x)
    x = layers.Conv2D(384, (3, 3),activation='relu',padding='same',name='conv3')(x)
    x = layers.Conv2D(384, (3, 3),activation='relu',padding='same',name='conv4')(x)
    x = layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='conv5')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='valid',name='pool5')(x)

    x = layers.Flatten()(x)
    #x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(4096,activation='relu',name='fc6')(x)
    x = layers.Dropout(0.5,name='drop6')(x)
    x = layers.Dense(4096,activation='relu',name='fc7')(x)
    x = layers.Dropout(0.5,name='drop7')(x)
    x = layers.Dense(classes,activation='softmax',name='fc1000')(x)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='alexnet')
    #model.summary()
    return model

    
