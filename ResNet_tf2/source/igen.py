# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import math
from tqdm import tqdm

# igenモジュール用Exception
class ImageGeratorException(Exception):
    pass


class ImageGenerator():
    def __init__(self, lines=None, isTrain=True, bs=None, nb_classes=None, mean_image=None):
        """各メンバの初期化
        lines: txtファイルの内容
        isTrain: 学習データ or 検証データ
        bs: バッチサイズ
        nb_classes: 分類クラス数
        mean_image: 平均画像
        """
        self.isTrain = isTrain # 学習データ or 検証データ
        self.image_paths = [] # 画像のパス
        self.image_labels = [] # 教師値
        self.bs = bs # バッチサイズ
        self.nb_classes = nb_classes # 分類クラス数
        self.mean_image = None # 平均画像
        self.imread = cv2.imread

        # パス，教師値をセット
        for line in lines:
            list = line.split()
            self.image_paths.append(list[0])
            self.image_labels.append(list[1])

        #平均画像をセット
        # 検証データにおいて平均画像がセットされていない場合，例外
        try:
            if isTrain and mean_image is None:
                self._create_mean_image()
            elif not isTrain and mean_image is None:
                raise ImageGeratorException()
            else:
                self.mean_image = mean_image
        except ImageGeratorException:
            print(ImageGeratorException)
            print('Please set a mean_image.')

    def _create_mean_image(self):
        """平均画像を生成する
        """
        if self.isTrain:
            data = np.zeros((1,256,256,3), dtype='uint8') # 1件分の画像
            mean_image = np.zeros((1,256,256,3), dtype='float32') # 平均画像用np配列
            for ipath in tqdm(self.image_paths):
                #print(ipath)
                data[0] = self.imread(ipath, cv2.IMREAD_COLOR)
                X_train = data.astype('float32')
                mean_image += X_train
            mean_image /= len(self.image_paths) #平均画像を作成
            self.mean_image = mean_image
        else:
            print("Can't create a mean image because this is a test dataset.")
            exit()

    def generate_batch(self, idx):
        """指定のバッチに相当する画像，ラベルを渡す
        idx: generatorからの引数，入力するバッチの添え字
        """
        request_paths = self.image_paths[idx*self.bs : (idx+1)*self.bs]
        request_labels = self.image_labels[idx*self.bs : (idx+1)*self.bs]
        images = np.zeros((len(request_paths),256,256,3), dtype='uint8')
        labels = np.zeros(len(request_labels), dtype='uint8')

        # 画像，ラベルを読み込む
        for i, (path, label) in enumerate(zip(request_paths, request_labels)):
            images[i,:,:,:] = self.imread(path, cv2.IMREAD_COLOR)
            labels[i] = int(label)

        # ネットワーク入力用に変換
        re_labels = to_categorical(labels, self.nb_classes)
        re_images = images.astype('float32')
        re_images -= self.mean_image
        re_images /= 128.
        return (re_images, re_labels)

    def get_batch_num(self):
        """バッチ数を返却
        """
        return math.ceil((len(self.image_paths) / self.bs))

    def get_mean_image(self):
        """平均画像を返却
        """
        return self.mean_image

    def get_paths(self):
        """全てのパスを返却
        """
        return self.image_paths
