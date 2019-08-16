# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :       yandongwei
   Date：          2018-12-19 下午2:33
   Description :
   
-------------------------------------------------
   Change Activity:

-------------------------------------------------
"""
# coding=utf-8
import os
import sys

import importlib

importlib.reload(sys)
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for
import uuid
import datetime
import tensorflow as tf
# from classify_image_inception_v3 import run_inference_on_image
from datetime import timedelta
from classify_image_inception_v3_copy import run_inference_on_image_copy
from setting import *

ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

FLAGS = tf.app.flags.FLAGS

"""Namespace(image_file='test_image.jpg', 
		label_path='data_prepare/pic/label.txt',
		model_path='slim/satellite/frozen_graph.pb', 
		num_top_predictions=5)"""

# tf.app.flags.DEFINE_string('model_path', '/home/ubutnu/chenjf/slim/satellite/frozen_graph.pb', """*****, """)
# tf.app.flags.DEFINE_string('label_path', '/home/ubutnu/chenjf/slim/satellite/data/labels.txt', '')
tf.app.flags.DEFINE_string('upload_folder', '/home/ubutnu/work/download_image/mytest', '')
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_integer('port', '8080',
                            'server with port,if no port, use deault port 80')

tf.app.flags.DEFINE_boolean('debug', True, '')

UPLOAD_FOLDER = FLAGS.upload_folder

app = Flask(__name__)
# app._static_folder = UPLOAD_FOLDER
app.send_file_max_age_default = timedelta(seconds=1)

def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    # new_name = str(uuid.uuid1()) + ext
    new_name = str(time.time()) + ext
    return new_name


def inference(file_name):
    try:
        # model_path = '/home/ubutnu/chenjf/data0/frozen_graph.pb'
        # label_path = '/home/ubutnu/chenjf/data0/label.txt'
        # image_file = '/home/ubutnu/work/download_image/'
        model_path = Model_path
        label_path = Label_path
        image_file = Image_file
        predictions = run_inference_on_image_copy(model_path, label_path, image_file, file_name, num_top_predictions1=1)
        # print('##################')
        # print(score)
        # print(type(predictions))
        # print(predictions)
    except Exception as ex:
        print(ex)
        return ""
    # new_url = '/static/%s' % os.path.basename(file_name)
    # image_tag = '<img src="%s"></img><p>'
    # new_tag = image_tag % new_url
    # format_string = ''
    # for node_id, human_name in zip(top_k, top_names):
    #     score = predictions[node_id]
    #     format_string += '%s (score:%.5f)<BR>' % (human_name, score)
    # ret_string = new_tag + format_string + '<BR>'
    # return ret_string
    return predictions
import cv2
import time
import shutil
from flask import Flask, render_template,url_for,make_response


@app.route("/show", methods=['GET'])
def show_web():
    return render_template('upload_ok.html')


@app.route("/infer", methods=['POST'])
def root():
    result = """
    <!doctype html>
    <title>测试版web服务</title>
    <h1>AI证件识别</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='选择图片'>
         <input type=submit value='上传'>
    </form>
    <p>%s</p>
    """ % "<br>"
    start_time=time.time()
    file = request.files['file']
    old_file_name = file.filename
    if file and allowed_files(old_file_name):
        filename = rename_filename(old_file_name)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        # type_name = 'N/A'
        # print('file saved to %s' % file_path)
        # img= cv2.imread(file_path)
        # print(img.shape)
        # if len(img.shpae)==3:
        # out_html = inference(file_path)
        # end_time=time.time()
        # spend_time=end_time-start_time
        # print(spend_time)
        basepath = os.path.dirname(__file__)
        #
        img_path = os.path.join(basepath, 'static/images', 'test.jpg')
        # print(file_path)
        #
        # file.save(file_path)

        shutil.copy(file_path,img_path)
        out_html = inference(file_path)

        # log_file = '/home/ubutnu/work/logs/log.txt'
        log_file=Log_file
        with open(log_file, 'a+') as f:
            if not os.path.exists(log_file):
                os.mknod(log_file)
            else:
                f.write(old_file_name + ','+file_path + ',' + out_html + '\r\n')
            f.close()
        return render_template('upload_ok.html', userinput=out_html)


    #         log_file = '/home/ubutnu/work/logs/log.txt'
    #         with open(log_file, 'a+') as f:
    #             if not os.path.exists(log_file):
    #                 os.mknod(log_file)
    #             else:
    #                 f.write(old_file_name + ','+file_path + ',' + out_html + '\r\n')
    #             f.close()
    #
    #         return result + str(out_html)
    # return result


@app.route("/test",methods=['GET'])
def test():
    return "service is running!"


if __name__ == "__main__":
    # set_flags(FLAGS)
    # print('listening on port %d' % FLAGS.port)
    # app.run(host='192.168.120.240', port=FLAGS.port, debug=FLAGS.debug, threaded=True)
    print('listening on port %s' % Port)
    app.run(host=Host, port=Port, debug=FLAGS.debug, threaded=True)
