# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2018-12-25
# File:        image_classify_inception.py
# Description: 生成slim数据格式的tfRecord
# 每个文件夹放一个类别图片即可
import os

dataset_name = 'flowers'
dataset_dir = '/home/ubutnu/tf-slim-inception/data/flowers'
command = 'python /home/ubutnu/tf-slim-inception/models-master/research/slim/download_and_convert_data.py ' \
          '--dataset_name=%s ' \
          '--dataset_dir=%s' % (dataset_name, dataset_dir)
os.system(command)



