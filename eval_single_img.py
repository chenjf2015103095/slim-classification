# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :       yandongwei
   Date：          2019-01-17 上午11:20
   Description :
   
-------------------------------------------------
   Change Activity:

-------------------------------------------------
"""

#
# import numpy as np
# from six.moves import urllib
# import tensorflow as tf
#
#
# class EvalNode(object):
#
#     def __init__(self, image_file, model_path, label_path,
#           image_size, num_top_predictions, output_node_name):
#         self.image_file = image_file
#         self.model_path = model_path
#         self.label_path = label_path
#         self.image_size = image_size
#         self.num_top_predictions = num_top_predictions
#         self.output_node_name = output_node_name
#         self.node_id_to_name = None
#
#     def load(self):
#         node_id_to_name = {}
#         with open(self.label_path) as f:
#             for index, line in enumerate(f):
#                 node_id_to_name[index] = line.strip()
#         self.node_id_to_name = node_id_to_name
#
#     def id_to_string(self, node_id):
#         if node_id not in self.node_id_to_name:
#             return ''
#         return self.node_id_to_name[node_id]
#
#
# def create_graph(node_lookup):
#     """Creates a graph from saved GraphDef file and returns a saver."""
#     # Creates graph from saved graph_def.pb.
#     with tf.gfile.FastGFile(node_lookup.model_path, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         _ = tf.import_graph_def(graph_def, name='')
#
#
# def preprocess_for_eval(image, height, width,
#                         central_fraction=0.875, scope=None):
#     with tf.name_scope(scope, 'eval_image', [image, height, width]):
#         if image.dtype != tf.float32:
#             image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#         # Crop the central region of the image with an area containing 87.5% of
#         # the original image.
#         if central_fraction:
#             image = tf.image.central_crop(image, central_fraction=central_fraction)
#
#         if height and width:
#             # Resize the image to the specified height and width.
#             image = tf.expand_dims(image, 0)
#             image = tf.image.resize_bilinear(image, [height, width],
#                                              align_corners=False)
#             image = tf.squeeze(image, [0])
#         image = tf.subtract(image, 0.5)
#         image = tf.multiply(image, 2.0)
#         return image
#
#
# def infer(image_file, model_path, label_path,
#           image_size=299, num_top_predictions=5, output_node_name='InceptionV4/Logits/Predictions:0'):
#     with tf.Graph().as_default():
#         image_data = tf.gfile.FastGFile(image_file, 'rb').read()
#         image_data = tf.image.decode_jpeg(image_data)
#         image_data = preprocess_for_eval(image_data, image_size, image_size)
#         image_data = tf.expand_dims(image_data, 0)
#         with tf.Session() as sess:
#             image_data = sess.run(image_data)
#     node_lookup = EvalNode(image_file, model_path, label_path, image_size, num_top_predictions, output_node_name)
#     create_graph(node_lookup)
#     with tf.Session() as sess:
#         softmax_tensor = sess.graph.get_tensor_by_name(node_lookup.output_node_name)
#         predictions = sess.run(softmax_tensor, {'input:0': image_data})
#         predictions = np.squeeze(predictions)
#         node_lookup.load()
#         top_k = predictions.argsort()[-node_lookup.num_top_predictions:][::-1]
#         result = {}
#         for node_id in top_k:
#             human_string = node_lookup.id_to_string(node_id)
#             score = predictions[node_id]
#             result[human_string] = score
#     return sorted(result.items(), key=lambda d: d[1], reverse=True)
#
# import os
# import cv2
# from PIL import Image
# if __name__ == '__main__':
#     files_path = '/home/ubutnu/work/download_image/myIDcard'
#     for root, dirs, files in os.walk(files_path):
#         # print(files)
#         for file in files:
#             # print(file)
#             file_path = os.path.join(root, file)
#             # print(file_path)
#     # file_path = '/home/ubutnu/work/download_image/test07.jpg'
#             num_top_predictions = 5
#             model_path = '/home/ubutnu/chenjf/IDcar/frozen_graph.pb'
#             label_path = '/home/ubutnu/chenjf/IDcar/label.txt'
#             image_size = 299
#             output_node_name = 'InceptionV3/Logits/SpatialSqueeze:0'
#             result = infer(file_path, model_path, label_path,
#                                        image_size, num_top_predictions, output_node_name)
#
#             # img=Image.open(file_path)
#             # new_path1='/home/ubutnu/work/download_image/damage/'
#             # new_path2='/home/ubutnu/work/download_image/whole/'
#             # new_path3='/home/ubutnu/work/download_image/others/'
#             print(result[0][0])
#             # if str(result[0][0])=='damage':
#             #     print('This is damage')
#
#             # if result[0][1]>10:
#             #     print('This is car!')
#
#                 # cv2.imwrite('/home/ubutnu/work/download_image/Car/file_path.jpg',file_path)
#
#             #     img.save(new_path1+file)
#             # elif str(result[0][0])=='others':
#             #     print('This is others')
#             #     img.save(new_path2+file)
#             # else:
#             #     img.save(new_path3+file)
#             print(result)
#











import numpy as np
from six.moves import urllib
import tensorflow as tf


class EvalNode(object):

    def __init__(self, model_path, label_path,
          image_size, num_top_predictions, output_node_name):
        self.model_path = model_path
        self.label_path = label_path
        self.image_size = image_size
        self.num_top_predictions = num_top_predictions
        self.output_node_name = output_node_name
        self.node_id_to_name = None
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.load()
        self.conf = tf.ConfigProto()
        self.conf.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.conf.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.gfile.FastGFile(self.model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph, config=self.conf)

    def load(self):
            node_id_to_name = {}
            with open(self.label_path) as f:
                for index, line in enumerate(f):
                    node_id_to_name[index] = line.strip()
            self.node_id_to_name = node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_id_to_name:
            return ''
        return self.node_id_to_name[node_id]

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def infer(self, image_file):
        result = {}
        msg={}
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        image_data = tf.image.decode_jpeg(image_data)
        image_data = preprocess_for_eval(image_data, self.image_size, self.image_size)
        image_data = tf.expand_dims(image_data, 0)
        with tf.Session() as sess:
            image_data = sess.run(image_data)
        softmax_tensor = self.sess.graph.get_tensor_by_name(self.output_node_name)
        predictions = self.sess.run(softmax_tensor, {'input:0': image_data})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-self.num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = self.id_to_string(node_id)
            score = predictions[node_id]
            msg[human_string] = score
        return sorted(msg.items(), key=lambda d: d[1], reverse=True)

        #     for node_id in top_k:
        #         human_string = self.id_to_string(node_id)
        #         result[human_string] = predictions[node_id]
        #     # print('%s (score = %.5f)' % (human_string, score))
        #
        # score= sorted(result.items(), key=lambda d: d[1], reverse=True)[0:1]



def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image





import os
import shutil
if __name__ == '__main__':
    files_path = '/home/ubutnu/work/download_image/myIDcard'
    num_top_predictions = 5
    # model_path = '/media/ubutnu/fc1a3be7-9b03-427e-9cc9-c4b242cefbff/chenjf_project/my_data/slim/satellite/frozen_graph.pb'
    # label_path = '/media/ubutnu/fc1a3be7-9b03-427e-9cc9-c4b242cefbff/chenjf_project/my_data/slim/satellite/data/label.txt'
    model_path = '/media/ubutnu/fc1a3be7-9b03-427e-9cc9-c4b242cefbff/tf-slim-inception/my_models/inception_v3/flowers/export/19068/frozen_inference_graph.pb'
    label_path = '/media/ubutnu/fc1a3be7-9b03-427e-9cc9-c4b242cefbff/tf-slim-inception/my_models/inception_v3/flowers/labels.txt'
    image_size = 299
    output_node_name = 'InceptionV3/Logits/SpatialSqueeze:0'
    node_lookup = EvalNode(model_path, label_path, image_size, num_top_predictions, output_node_name)
    # for root, dirs, files in os.walk(files_path):
    #     # print(files)
    #     for file in files:
    #         # print(file)
    #         file_path = os.path.join(root, file)
    #         # print(file_path)
    #         result = node_lookup.infer(file_path)
    #         print(result)
    #         new_path1 = '/home/ubutnu/work/three/last/damage/'
    #         new_path2 = '/home/ubutnu/work/three/last/others/'
    #         new_path3 = '/home/ubutnu/work/three/last/whole/'
    #         if result is None:
    #             continue
    #         print('Load success!')
    #         print(result[0][0])
    #         if str(result[0][0]) == 'damage':
    #             print('This is damage')
    #             shutil.move(file_path,new_path1)
    #         elif str(result[0][0]) == 'others':
    #             print('This is others')
    #             shutil.move(file_path, new_path2)
    #         else:
    #             print('This is whole!')
    #             shutil.move(file_path, new_path3)
    #         print(result)

