# -*- coding: utf-8 -*-
# update: 20/02/13
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import optimizers
#import tensorflow.keras.backend as K
import numpy as np
from iread import ImageReader
from igen import ImageGenerator
from iseq import ImageSequence
import multiprocessing
import os
import random as rd
import configparser

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")
tf.random.set_seed(0)

"""
def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))
"""

def create_model(network='resnet18',input_shape=None, pooling=None, classes=3):
    model = None
    if network == 'resnet18':
        from network_models import resnet18
        model = resnet18.ResNet18(input_tensor=None,
                                  input_shape=input_shape,
                                  pooling=None,
                                  classes=classes)
    elif network == 'resnet18v2':
        from network_models import resnet18v2
        model = resnet18v2.ResNet18v2(input_tensor=None,
                                  input_shape=input_shape,
                                  pooling=None,
                                  classes=classes)
    elif network == 'resnet50':
        from network_models import resnet50
        model = resnet50.ResNet50(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif network == 'resnet50v2':
        from network_models import resnet50v2
        model = resnet50v2.ResNet50v2(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif network == 'resnet101':
        from network_models import resnet101
        model = resnet101.ResNet101(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif network == 'resnet101v2':
        from network_models import resnet101v2
        model = resnet101v2.ResNet101v2(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif network == 'vgg19':
        from network_models import vgg19
        model = vgg19.VGG19(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif network == 'vgg16':
        print("using model :vgg16")
        from network_models import vgg16
        model = vgg16.VGG16(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif network == 'alexnet':
        print("using model :alexnet")
        from network_models import alexnet
        model = alexnet.AlexNet(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    elif network == 'efficientnet':
        print("using model :efficientnet")#なんかしないと使えまへん
        from network_models import efficientnet
        model = efficientnet.EfficientNet(input_tensor=None,
                                input_shape=input_shape,
                                pooling=None,
                                classes=classes)
    else:
        print(f"Not prepare model type \"{network}\"")
        exit()
    return model

def create_optimzer(opt, lr):
    if opt == 'SGD':
        return optimizers.SGD(lr=lr, momentum=0.9)
    elif opt == 'Adam':
        return optimizers.Adam(lr=lr)
    else:
        print('Not defined {}'.format(opt))
        exit()

def make_dirs(valname, savefiles, root='../products', date=None):
    """各種ファイルの保存先を生成
    valname: 検証用txtの識別番号(ex. 3n100_01)
    dir_paths: 保存先となるディクレクトリのパス
    """
    save_locate, _ = valname.rsplit('_', 1)
    save_locate = f'product_{save_locate}' # ex. product_3n100
    if date is not None:
        save_locate = f'{save_locate}_{date}'
    save_locate = os.path.join(root, save_locate)
    paths_dic = {'save_root' : save_locate}
    # 保存先のディレクトリがなければ作成
    for savefile in savefiles:
        dir_path = savefile if savefile != '_wh5' else f"{valname}{savefile}"
        dir_path = os.path.join(save_locate, dir_path)
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        paths_dic[savefile] = dir_path
    return paths_dic

def run(val, conf, meanimg=None, date=None):
    """ネットワークの設定，学習を行う．
    Args:
        val (str): 交差検証にて検証に用いるテキストファイルのパス．
        meanimg (str): 各パラメータが記述された設定ファイル．
        meanimg (str): 平均画像のパス．セットされてなければ，ImageGeneratorで作成する．
        date (str): 実行時刻
    """
    """ config file の読み込み """
    config = configparser.ConfigParser()
    config.read(conf)
    """ 各種ネットワークの設定 """
    _sec1 = 'params'
    train_bs = config.getint(_sec1, 'train_bs') # 学習用バッチサイズ
    test_bs = config.getint(_sec1, 'test_bs') # 検証用バッチサイズ
    nb_epoch = config.getint(_sec1, 'nb_epoch') #エポック数
    workers = config.getint(_sec1, 'workers') # 同時実行プロセス数，ジェネレータ用の設定
    max_queue_size = config.getint(_sec1, 'max_queue_size') # データ生成処理キューイング限度
    img_rows = config.getint(_sec1, 'img_rows') # 入力画像横幅
    img_cols = config.getint(_sec1, 'img_cols') # 入力画像縦幅
    img_channels = config.getint(_sec1, 'img_channels') # チャンネル数(RGB: 3)
    data_augmentation = config.getboolean(_sec1, 'data_augmentation') # 画像の水増し
    nb_classes = config.getint(_sec1, 'nb_classes') # 分類クラス数
    loss_function = 'binary_crossentropy' if nb_classes == 2 else 'categorical_crossentropy'
    network = config.get(_sec1, 'network') # 使用するCNNモデル
    lr = config.getfloat(_sec1, 'lr') # 最適化関数
    optimizer = config.get(_sec1, 'optimizer') # 最適化関数
    gpu_serial = config.get(_sec1, 'gpu_serial') # 使用するGPUのシリアル番号
    """ コールバック設定 """
    _sec2 = 'callbacks'
    use_lr_reducer = config.getboolean(_sec2, 'lr_reducer')
    use_early_stopper = config.getboolean(_sec2, 'early_stopper')
    use_model_checkpoint = config.getboolean(_sec2, 'model_checkpoint')
    """ 生成ファイル保管場所を設定 """
    _sec3 = 'savefiles'
    savefiles = (config.get(_sec3, i) for i in config.options(_sec3))
    """ 保存先のルートを設定 """
    root = config.get('savelocate', 'root')
    print(val)
    _valname, _ = os.path.basename(val).rsplit('.', 1) # 保存するファイル名に使用
    paths_dic = make_dirs(_valname, savefiles, root=root, date=date)
    """ 実験に使用した設定ファイルの保管 """
    _conf_locate = os.path.join(paths_dic['save_root'], os.path.basename(conf))
    if not os.path.exists(_conf_locate):
        import shutil
        shutil.copy(conf, paths_dic['save_root'])

    """ コールバックの設定 """
    callbacks = []
    """ 評価値改善が止まった時，学習率を減らす．"""
    # factor: 減少率(√0.1)
    # cooldown: 学習率を減らした後，通常の学習を再開するまで待機するエポック数(0)
    # min_lr: 学習率の下限(0.0000005)
    if use_lr_reducer:
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks.append(lr_reducer)
    """ 監視する値が変化停止すると，訓練を終了する．"""
    # min_delta: 監視値が改善されているか判定する閾値．監視値の絶対値がmin_deltaより小さければ改善していないとみなす．
    # patience: 訓練が停止し，値が改善しなくなってからのエポック数(10)
    if use_early_stopper:
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        callbacks.append(early_stopper)
    """ 最良の学習結果を保存する．(重みのみ) """
    if use_model_checkpoint:
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(paths_dic['_wh5'], ''.join((_valname, '_{epoch:02d}.h5'))),
            #monitor='val_get_categorical_accuracy_keras',
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        callbacks.append(model_checkpoint)
    """ csvにて結果を出力．(epock, acc，loss，lr，val_acc，val_loss)"""
    csv_logger = CSVLogger(os.path.join(paths_dic['log'], f"{_valname}_log.csv"))
    callbacks.append(csv_logger)

    """ データセット呼び出しの準備 """
    image_reader = ImageReader()
    (train_ig, test_ig) = image_reader.create_generators(val, train_bs, test_bs, nb_classes, meanimg)
    image_reader.save_mean_image(train_ig, _valname, paths_dic['mean'])

    """ モデルのロード"""
    """
    # weight(ネットワーク構成付き) からロード
    from keras.models import load_model
    model = load_model(args[3])
    """
    """
    # json + weight からロード
    json_string = open(args[3]).read()
    model = model_from_json(json_string)
    model.load_weights(args[4])
    """

    with tf.device(f'/device:GPU:{gpu_serial}'):
        """ モデルの定義 """
        #(入力画像データ(3,256,256),クラス数 3)
        model = create_model(network=network, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes)
        #モデルを設定(損失、最適化アルゴリズム、訓練時とテスト時に評価される関数リスト)
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[get_categorical_accuracy_keras])
        #model.compile(loss=loss_function, optimizer=optimizer, metrics=['acc'])
        model.compile(loss=loss_function, optimizer=create_optimzer(optimizer, lr), metrics=['acc'])

        model.summary()
        """ ネットワークの可視化．python3.5以降では利用できないので変更が必要 """
        # plot_model(model, to_file='model.png', show_shapes=True)

        """ 実行 """
        print('Not using data augmentation.')
        model.fit_generator(
            generator=ImageSequence(train_ig),
            steps_per_epoch=train_ig.get_batch_num(),
            epochs=nb_epoch,
            verbose=1, # verbose : '1'にすると，実行経過を棒グラフで表示する
            #callbacks=[lr_reducer,early_stopper,csv_logger,model_checkpoint],
            callbacks=callbacks,
            validation_data=ImageSequence(test_ig),
            validation_steps=test_ig.get_batch_num(),
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=True,
            shuffle=False)

    #モデルのセーブ
    json_string = model.to_json()
    open(os.path.join(paths_dic['json'], f"{_valname}.json"), 'w').write(json_string)
    # モデルの構造，重み，学習の設定，optimizerの状態を保存
    #model.save(os.path.join(paths_dic['h5'], f"{_valname}.h5"))
    # 重みのみの保存
    model.save_weights(os.path.join(paths_dic['wh5'], f"{_valname}_w.h5"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='start to train a network by using resnet.py')
    parser.add_argument('val', help='path of a validation text')
    parser.add_argument('conf', help='config file writed params')
    parser.add_argument('--mean', '-m', default=None, help='path of a mean image')
    parser.add_argument('--datetime', '-d', default=None, help='start time')
    args = parser.parse_args()
    run(args.val, args.conf, args.mean, args.datetime)
