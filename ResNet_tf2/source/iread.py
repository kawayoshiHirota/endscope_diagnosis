# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import random as rd
import argparse
import re
from igen import ImageGenerator

# ireadモジュール用Exception
class ImageReadException(Exception):
    pass


class ImageReader():
    """txtファイルを元に，データセットを形成する．
    平均画像の保存を行う．
    インスタンス生成後，create_generators()を行うこと．
    """
    def __init__(self):
        pass

    def create_generators(self, val=None, train_bs=None, test_bs=None, nb_classes=None, mi_path=None):
        """iseq用に学習用・検証用のImageGeneratorのインスタンスを返す
        Args:
            val str: 検証用txtのパス．
            train_bs int: 学習時のバッチサイズ
            test_bs int: 検証時のバッチサイズ
            nb_classes int: 分類クラス数
            mi_path str: 平均画像のパス
        """
        # 事前に平均画像を用意している場合はセット
        mean_image = None if mi_path is None else np.load(mi_path)

        # 学習用ImageGeneratorの生成
        filelist = sorted(glob.glob(os.path.dirname(val) + '/*.txt'))
        lines = [] # テキストの内容を集める
        for file in filelist:
            if not val in file:
                lines += open(file,'r')
        rd.seed(0)
        rd.shuffle(lines)
        train_ig = ImageGenerator(lines, True, train_bs, nb_classes, mean_image)

        # 検証用ImageGeneratorの生成
        f = open(val, 'r')
        lines = f.read().splitlines()
        f.close()
        # 検証用に平均画像を取得
        if mean_image is None: mean_image = train_ig.get_mean_image()
        test_ig = ImageGenerator(lines, False, test_bs, nb_classes, mean_image)

        return (train_ig, test_ig)

    def save_mean_image(self, ImageGenerator, val_num, output):
        """平均画像を保存する
        ex) [output]/mean_3N100_01.npy
        Args:
            ImageGenerator: 学習用ImageGenerator，メンバのmean_imageを保存
            output str: 平均画像の保存先
        """
        try:
            if os.path.exists(output):
                np.save(os.path.join(output, val_num + '_mean.npy'), ImageGenerator.mean_image)
            else:
                raise ImageReadException()
        except ImageReadException:
            print("ImageReadException")
            print("Can't save a mean image")


if __name__ == '__main__':
    """平均画像を作成，保存する
    """
    parser = argparse.ArgumentParser(description="make a mean piture ")
    parser.add_argument("val", help="txt path for validation")
    parser.add_argument("output", help="output locate path for a mean image")
    args = parser.parse_args()

    image_reader = ImageReader()
    (train_ig, test_ig) = image_reader.create_generators(val=args.val)
    val_num = re.sub('\..*', '', os.path.basename(args.val))
    image_reader.save_mean_image(train_ig, val_num, args.output)
