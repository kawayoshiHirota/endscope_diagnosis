# -*- coding: utf-8 -*-
# update: 19/10/15
import configparser

# 設定ファイルを生成するモジュール
def create_config(name='config'):
    config = configparser.ConfigParser()

    section1 = 'params'
    config.add_section(section1)
    config.set(section1, 'train_bs', '32')
    config.set(section1, 'test_bs', '32')
    config.set(section1, 'nb_epoch', '50')
    config.set(section1, 'workers', '4')
    config.set(section1, 'max_queue_size', '10')
    config.set(section1, 'img_rows', '256')
    config.set(section1, 'img_cols', '256')
    config.set(section1, 'img_channels', '3')
    config.set(section1, 'data_augmentation', 'false')
    config.set(section1, 'nb_classes', '2')
    config.set(section1, 'network', 'resnet18')
    config.set(section1, 'lr', '0.0001')
    config.set(section1, 'optimizer', 'SGD')
    config.set(section1, 'gpu_serial', '0')

    section2 = 'callbacks'
    config.add_section(section2)
    config.set(section2, 'lr_reducer', 'true')
    config.set(section2, 'early_stopper', 'false')
    config.set(section2, 'model_checkpoint', 'true')

    section3 = 'savefiles'
    config.add_section(section3)
    config.set(section3, 'item1', 'json')
    #config.set(section3, 'item2', 'h5')
    config.set(section3, 'item3', 'wh5')
    config.set(section3, 'item4', 'log')
    config.set(section3, 'item5', 'mean')
    config.set(section3, 'item6', '_wh5')

    section4 = 'savelocate'
    config.add_section(section4)
    config.set(section4, 'root', '../products')

    with open(f'{name}.ini', 'w') as file:
        config.write(file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='create a config file of network')
    parser.add_argument('--name', '-n', default='config', help='config file name')
    args = parser.parse_args()
    create_config(args.name)
