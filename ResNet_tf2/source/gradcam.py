# -*- coding:utf-8 -*-
import numpy as np
import cv2
import PIL
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.models import load_model, model_from_json
import tensorflow as tf
import os
import glob

# 19/11/21

K.set_learning_phase(1)

IMAGE_SIZE=(256, 256)

def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

def Grad_Cam(model, x, layer_name):
    '''
    Args:
        model: model_object(keras.models.Model)
        x: pictuer( array )
        layer_name: conv2d layer name (check by model.summary() )

    Returns:
        jetcam: picture array
    '''
    # preprocessing
    X = np.expand_dims(x, axis=0)
    x = X.astype("float32")
    preprocessed_input = X / 255.0

    # predict class
    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    # Get Grad
    conv_output = model.get_layer(layer_name).output
    #grads = K.gradients(class_output, conv_output)[0]
    # 上記では，tf-2.0に対応していないため，以下で勾配情報を取得
    g = tf.Graph()
    with g.as_default():
        grads = tf.gradients(class_output, conv_output)[0]
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # average and dot
    weights = np.mean(grads_val, axis=(0,1))
    cam = np.dot(output, weights)

    # Generate Heatmap
    cam = cv2.resize(cam, IMAGE_SIZE, cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    jetcam = np.maximum(jetcam, cv2.COLOR_BGR2RGB)
    jetcam = (np.float32(jetcam) + x / 2)

    return jetcam

def main(json, weight, target, layer_name, root):
    # モデル読み込み
    model = model_from_json(open(json).read())
    model.load_weights(weight)

    # 保存場所
    uniq, _ = os.path.basename(json).rsplit('.', 1)
    dir_name = '{}_{}'.format(uniq, layer_name)
    dir_name = os.path.join(root, dir_name)
    if not os.path.exists(dir_name): os.makedirs(dir_name)

    # 対象画像をリスト化
    img_paths = []
    if 'jpg' in target: # target: 画像
        img_paths.append(target)
    else: # target: 画像を内包するディレクトリ
        img_paths = sorted(glob.glob(f'{target}/*.jpg'))

    # 画像群に対しGradCAMを実行
    for img_path in img_paths:
        print(img_path)
        x = img_to_array(load_img(img_path, target_size=IMAGE_SIZE))
        image = Grad_Cam(model, x, layer_name)
        image = image.reshape((256, 256, 3))
        #print(image.shape)
        x =  cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        img_name = os.path.join(dir_name, os.path.basename(img_path))
        cv2.imwrite(img_name, cv2.hconcat([x, image]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('json', help='json')
    parser.add_argument('weight', help='weight file')
    parser.add_argument('target', help='target image / dir inculude image')
    parser.add_argument('--layer', '-l', default="activation_16", help='name of target layer')
    parser.add_argument('--root', '-r', default="./test_gradcam_out", help='output file root')
    args = parser.parse_args()

    main(args.json, args.weight, args.target, args.layer, args.root)
