#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
before start testing
set my_train_mode in ./lib/datasets/pascal_voc.py and ./tools/demo.py to False
"""

my_train_mode = False

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from model.config import cfg
from model.test import im_detect

from torchvision.ops import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

import os.path as osp
import os
from time import time as tm
import time
from datachange import DataChange
from draw_rectangle import DrawRectangle

dataname ='2019_5000m'
rootpath = 'E:/chLi/pytorch-faster-rcnn'


datarootpath = rootpath + '/data/' + dataname
source_image_path = datarootpath + '/source/JPEGImages'
source_json_path = datarootpath + '/source/Annotations'

need_to_change_data = True
use_my_labels = False

need_to_change_image_size = False
image_width = 426
image_height = 240

need_to_evaluate_data = True

need_to_draw_rectangle = True
rectangle_color = [0, 0, 255]
rectangle_width = 5

need_to_show_name = True
name_color = (0, 255, 0)

need_to_evaluate_result = True

if my_train_mode:
    need_to_change_image_size = True
    need_to_evaluate_data = False
    need_to_draw_rectangle = False
    need_to_show_name = False
    need_to_evaluate_result = False

model_list = os.listdir('E:/chLi/pytorch-faster-rcnn/output/default/voc_2007_trainval/default')
resume_num = 0
for model_name in model_list:
    if int(model_name.split('.')[0].split('_')[4]) > resume_num:
        resume_num = int(model_name.split('.')[0].split('_')[4])
resume_num = 0
if resume_num == 0:
    resume_name = rootpath + '/output/res101/voc_2007_trainval+voc_2012_trainval/default/res101_faster_rcnn_iter_110000.pth'
else:
    resume_name = rootpath + '/output/default/voc_2007_trainval/default/res101_faster_rcnn_iter_%d.pth' % resume_num

net_name = resume_name.split('/output/')[1].split('/')[3].split('.pth')[0]

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

NETS = {
    'vgg16': ('vgg16_faster_rcnn_iter_%d.pth', ),
    'res101': ('res101_faster_rcnn_iter_%d.pth', )
}
DATASETS = {
    'pascal_voc': ('voc_2007_trainval', ),
    'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval', )
}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False,
                          edgecolor='red',
                          linewidth=3.5))
        ax.text(
            bbox[0],
            bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14,
            color='white')

    ax.set_title(
        ('{} detections with '
         'p({} | box) >= {:.1f}').format(class_name, class_name, thresh),
        fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    f_r = open(datarootpath + '/result_data.txt', 'r')
    lines = f_r.readlines()
    f_r.close()

    insert_idx = 0
    while 'GROUND TRUTH FOR: ' + image_name.split('.')[0] not in lines[insert_idx]:
        insert_idx += 1
    while lines[insert_idx] != '\n':
        insert_idx += 1
        if insert_idx == len(lines):
            break

    f_w = open(datarootpath + '/result_data.txt', 'w')
    for i in range(insert_idx):
        f_w.write(lines[i])

    f_w.write('PREDICTIONS: \n')

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = os.path.join(source_image_path, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    # timer = Timer()
    # timer.tic()
    scores, boxes = im_detect(net, im)
    # timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(
    #     timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    time3_sum = 0
    label_idx = 0
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) > 0:
            for i in inds:
                label_idx += 1
                bbox = dets[i, :4]
                f_w.write('%d label: ' % label_idx)
                f_w.write(cls)
                f_w.write(' score: tensor(?) ')
                f_w.write('%.4f || %.4f || %.4f || %.4f\n' % (bbox[0], bbox[1], bbox[2], bbox[3]))

        time3 = tm()
        DrawRectangle(datarootpath, net_name, im, cls, dets, CONF_THRESH, need_to_draw_rectangle, rectangle_color, rectangle_width, need_to_show_name, name_color, use_my_labels, need_to_evaluate_result)
        time3_sum += tm() - time3

    for i in range(insert_idx, len(lines)):
        f_w.write(lines[i])
    f_w.close()

    if not osp.exists(datarootpath + '/results'):
        os.mkdir(datarootpath + '/results')
    if not osp.exists(datarootpath + '/results/' + net_name):
        os.mkdir(datarootpath + '/results/' + net_name)

    cv2.imwrite(datarootpath + '/results/' + net_name + '/' + image_name, im)

    return time3_sum


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tensorflow Faster R-CNN demo')
    parser.add_argument(
        '--net',
        dest='demo_net',
        help='Network to use [vgg16 res101]',
        choices=NETS.keys(),
        default='res101')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Trained dataset [pascal_voc pascal_voc_0712]',
        choices=DATASETS.keys(),
        default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    total_time = tm()

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    if need_to_change_data:
        print('Start changing source data ...')
        time1 = tm()
        DataChange(datarootpath, source_json_path, source_image_path, use_my_labels, need_to_change_image_size, image_width, image_height)
        time1 = tm() - time1
        print('Finished changing source data!')

    if need_to_draw_rectangle or need_to_show_name:
        need_to_evaluate_data = True

    if need_to_evaluate_data:
        time2 = tm()

        f_copy_r = open(datarootpath + '/ground_truth.txt', 'r')
        content_copy = f_copy_r.read()
        f_copy_r.close()
        f_copy_w = open(datarootpath + '/result_data.txt', 'w')
        f_copy_w.write(content_copy)
        f_copy_w.close()

        # model path
        demonet = args.demo_net
        dataset = args.dataset
        # saved_model = os.path.join(rootpath +
        #                            '/output', demonet, DATASETS[dataset][0], 'default',
        #                            NETS[demonet][0] % (70000 if dataset == 'pascal_voc' else 110000))
        saved_model = resume_name

        if not os.path.isfile(saved_model):
            raise IOError(
                ('{:s} not found.\nDid you download the proper networks from '
                 'our server and place them properly?').format(saved_model))

        # load network
        print('Start loading model ...')
        if demonet == 'vgg16':
            net = vgg16()
        elif demonet == 'res101':
            net = resnetv1(num_layers=101)
        else:
            raise NotImplementedError
        net.create_architecture(21, tag='default', anchor_scales=[8, 16, 32])

        net.load_state_dict(
            torch.load(saved_model, map_location=lambda storage, loc: storage))

        net.eval()
        if not torch.cuda.is_available():
            net._device = 'cpu'
        net.to(net._device)

        print('Finished loading model!')

        print('Loaded network {:s}'.format(saved_model))

        # im_names = [
        #     '000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg'
        # ]

        print('Start evaluating data ...')
        time2 = tm()
        time3 = 0

        im_names = os.listdir(datarootpath + '/JPEGImages')
        idx = 0
        idx_total = len(im_names)

        for im_name in im_names:
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print('Demo for data/demo/{}'.format(im_name))
            time3 += demo(net, im_name)
            idx += 1
            print('process : %d / %d' % (idx, idx_total))

        time2 = tm() - time2 - time3
        print('Finished evaluating  data!')

    # plt.show()

    total_time = tm() - total_time

    f = open(datarootpath + '/output_msg.txt', 'a+')
    f.write('\n')
    f.write('===========================================\n')
    f.write('===========================================\n')
    f.write('===========================================\n\n')
    f.write('      ---- Date ----          : ' + time.asctime(time.localtime(tm())) + '\n\n')
    f.write('      ---- Net  ----          : ' + resume_name.split(rootpath)[1].split('/')[5] + '\n\n')
    f.write('-------------------------------------------\n')
    if need_to_change_data:
        print('Spending time on changing source data : %.2fms' % (time1 * 1000))
        f.write('Spending time on changing source data : %.2fms\n' % (time1 * 1000))
    if need_to_evaluate_data:
        print('Spending time on evaluating data : %.2fms' % (time2 * 1000))
        f.write('Spending time on evaluating data : %.2fms\n' % (time2 * 1000))
    if need_to_draw_rectangle and need_to_show_name:
        print('Spending time on drawing rectangle and showing name : %.2fms' % (time3 * 1000))
        f.write('Spending time on drawing rectangle and showing name : %.2fms\n' % (time3 * 1000))
    elif need_to_draw_rectangle:
        print('Spending time on drawing rectangle : %.2fms' % (time3 * 1000))
        f.write('Spending time on drawing rectangle : %.2fms\n' % (time3 * 1000))
    elif need_to_show_name:
        print('Spending time on showing name : %.2fms' % (time3 * 1000))
        f.write('Spending time on showing name : %.2fms\n' % (time3 * 1000))

    print('Spending time on total process : %.2fms' % (total_time * 1000))
    f.write('Spending time on total process : %.2fms\n' % (total_time * 1000))
    f.write('-------------------------------------------\n')
    f.close()

    if need_to_evaluate_result:
        DrawRectangle(datarootpath, need_to_evaluate_result=True)