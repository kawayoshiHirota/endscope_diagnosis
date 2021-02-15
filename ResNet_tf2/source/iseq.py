# -*- coding:utf-8 -*-
from tensorflow.keras.utils import Sequence
from igen import ImageGenerator

class ImageSequence(Sequence):
    def __init__(self, ImageGenerator):
        self.ImageGenerator = ImageGenerator # ImageGeneratorのインスタンス
        self.batch_num = self.ImageGenerator.get_batch_num()

    def __getitem__(self, idx):
        return self.ImageGenerator.generate_batch(idx)

    def __len__(self):
        # 分割データ数，idxの最大値はここから取っている
        return self.batch_num

    def on_epoch_end(self):
        # epoch終了時の処理
        pass
