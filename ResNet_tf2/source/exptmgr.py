# -*- coding: utf-8 -*-
# update: 19/10/15．
import os
import glob
from datetime import datetime
import monitor

def cross_validation(val_dir, conf, mean_dir=None):
    vals = sorted(glob.glob(f"{val_dir}/*.txt"))
    if not mean_dir:
        means = [ None for _ in vals]
    else:
        exist_means = sorted(glob.glob(f"{mean_dir}/*.npy"))
        if len(exist_means) == 0:
            print('Not validation file in {}'.format(mean_dir))
            exit()
        tags = [os.path.basename(var).rsplit('.', 1)[0] for var in vals]
        means = [mean if tag in mean else None for mean in exist_means for tag in tags]
    date = datetime.now().strftime("%Y%m%d%H%M") # 時刻を取得
    [monitor.run(val, conf, meanimg=mean, date=date) for val, mean in zip(vals, means)]

def test_validation(val, conf, mean):
    if mean and not '.npy' in mean:
        print('{} is not supported format as mean image'.format(mean))
        exit()

    import time
    start = time.time()
    date = datetime.now().strftime("%Y%m%d%H%M") # 時刻を取得
    monitor.run(val, conf, meanimg=mean, date=date)
    end = time.time()

    with open('./timerecord.txt', 'a') as f:
        f.write('{}\n'.format(date))
        f.write('training: {} sec\n'.format(end-start))

def main(target, conf, cv, mean):
        if cv:
            cross_validation(target, conf, mean)
        else:
            test_validation(target, conf, mean)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Start training')
    parser.add_argument('target', help='validation target')
    parser.add_argument('conf', help='config file')
    parser.add_argument('--cv', '-c', action='store_true', help='cross_validation mode')
    parser.add_argument('--mean', '-m', default=None, help='mean image')

    args = parser.parse_args()
    main(args.target, args.conf, args.cv, args.mean)
