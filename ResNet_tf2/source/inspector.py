# -*- coding: utf-8 -*-
from tensorflow.keras.models import model_from_json
import tensorflow as tf

import tensorflow.keras.backend as K
import numpy as np
#import multiprocessing
from igen import ImageGenerator
from iseq import ImageSequence
import re
import argparse
import os
import gc
import math
from tqdm import tqdm

# update: 20/01/10

def get_argument(weight, target):
    """ファイル・保存先のディレクトリをまとめた辞書を生成．辞書は関数runの引数として使用する．
        ディレクトリ構造に依存するため注意．
    Args:
        weight (str): 重みファイルのパス
        target (str): 推論対象となるtxt，形式:"[pass] [supervised value]"
    Returns:
        dict: runメソッドの使用する辞書
    """
    """重みファイルのパスから固有名を発見 ex)2n100_01"""
    weight_dir = os.path.basename(os.path.dirname(weight))
    if weight_dir == 'wh5':
        uniq = re.sub('_w\.h5', '', os.path.basename(weight))
    else:
        # ModelCheckpointの重みファイルを利用する場合
        uniq = re.sub('_wh5', '', weight_dir)

    """固有名からファイルパスを探索，生成し，辞書に登録"""
    arg_dict = {} # 返却値となる辞書
    product_root = os.path.dirname(os.path.dirname(weight))
    arg_dict["weight"] = weight
    arg_dict["json"] = os.path.join(product_root, f'json/{uniq}.json')
    arg_dict["mean"] = os.path.join(product_root, f'mean/{uniq}_mean.npy')
    arg_dict["target"] = target
    result_dir = os.path.join(product_root, 'result')
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    arg_dict["miss"] = os.path.join(result_dir, 'miss_' + os.path.basename(target))
    return arg_dict


def run(weight, target, classes=3):
    """推論を行う．誤分類したデータを纏めたmiss.txtを書き出す．
    Args:
        produtcs_path (str): monitoryでの生成物が含まれているディレクトリのパス
            ex). /home/matsui/p_res_v10/products/3n100_product
        cv_num (str): 交差検証での番号(01~10)
        test_file (str): テストに使用する画像のパス
    """
    arg_dict = get_argument(weight, target)
    mean = np.load(arg_dict['mean'])
    NB_CLASSES = classes # 識別クラス数
    BATCH_SIZE = 150 # テストデータのバッチサイズ
    IMAGE_NUM = 20000 # 1度に読み込む画像の枚数，メモリ対策
    count = [0] * NB_CLASSES # 各クラスにおいて分類された画像数を記録

    """ モデルのロード """
    model = model_from_json(open(arg_dict['json']).read())
    model.load_weights(arg_dict['weight'])
    #model.summary()
    loss_function = 'binary_crossentropy' if NB_CLASSES == 2 else 'categorical_crossentropy'
    model.compile(loss=loss_function,
                   optimizer='adam',
                   metrics=['acc'])

    """ テストファイルの読み込み """
    with open(arg_dict['target'],'r') as t:
        all_lines = t.read().splitlines()

    """ 誤分類の結果を書き込む """
    # [path] [supervised value] [predict value]
    with open(arg_dict['miss'], 'w') as g:
        iterate = math.ceil((len(all_lines) / IMAGE_NUM))
        data = np.zeros((1,256,256,3),dtype='float32')
        # メモリ対策の為，IMAGE_NUM分ごとにテスト
        for j in range(iterate):
            lines = all_lines[j*IMAGE_NUM : (j+1)*IMAGE_NUM]
            test_ig = ImageGenerator(lines, False, len(lines), NB_CLASSES, mean)
            (images, labels) = test_ig.generate_batch(0)
            for i, line in tqdm(enumerate(lines)):
                data[0,:,:,:] = images[i,:,:,:]
                y = model.predict(data)
                max_t = np.argmax(labels[i])
                max_p = np.argmax(y[0])
                if max_p != max_t:
                    g.write(f'{line} {str(max_p)}\n')
                count[max_p] += 1
            del images, labels
            gc.collect
    print(count)

    """
    # 評価
    test_ig = ImageGenerator(all_lines, False, BATCH_SIZE, NB_CLASSES, mean)
    score = model.evaluate_generator(generator=ImageSequence(test_ig))
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])
    # Attribute Error 対策，メモリ解放
    K.clear_session()
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='start to predict by using a generated network')
    parser.add_argument('wh5', help='h5 (only weight)')
    parser.add_argument('test', help='predicted text file')
    parser.add_argument('--classes', '-c', type=int, default=3, help='num of classes')
    #parser.add_argument('result', help='output file path to sava result')

    args = parser.parse_args()
    run(args.wh5, args.test, args.classes)
