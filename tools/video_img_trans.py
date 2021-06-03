import cv2
import os.path as osp
import os
from PIL import Image

rootpath = 'E:/chLi/pytorch-faster-rcnn'
data_dir = '2019_5000m'
video_name = '2019_5000m.mp4'
net_name = 'res101_faster_rcnn_iter_72000'

datarootpath = rootpath + '/data/' + data_dir
video_path = datarootpath + '/' + video_name
source_video_path = datarootpath + '/source'
source_image_path = datarootpath + '/source/JPEGImages'
source_json_path = datarootpath + '/source/Annotations'
result_image_path = datarootpath + '/results/' + net_name

if not osp.exists(source_video_path):
    os.mkdir(source_video_path)
if not osp.exists(source_image_path):
    os.mkdir(source_image_path)
if not osp.exists(source_json_path):
    os.mkdir(source_json_path)

class VideoImgTrans:

    def __init__(self):
        self.video_name = video_name
        self.video_path = video_path
        self.source_video_path = source_video_path
        self.source_image_path = source_image_path

    def video2img(self):

        vc = cv2.VideoCapture(self.video_path)
        c = 1
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        fps_catch = 5

        while rval:
            for i in range(fps_catch):
                if rval:
                    rval, frame = vc.read()
            cv2.imwrite(source_image_path + '/' + self.video_name.split('.')[0] + '_' + str(c) + '.jpg', frame)
            print(str(c) + '.jpg' + ' done!')
            c = c+1
            # cv2.waitKey(1)
        vc.release()

    def img2video(self):

        fps = 12
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        p = 1
        while not osp.exists(result_image_path + '/' + self.video_name.split('.')[0] + '_' + str(p) + '.jpg') and p < 10000:
            p += 1
        im = Image.open(result_image_path + '/' + self.video_name.split('.')[0] + '_' + str(p) + '.jpg')

        video_writer = cv2.VideoWriter(filename=datarootpath + '/result_' + video_name.split('.')[0] + '_' + net_name + '.' + video_name.split('.')[1], fourcc=fourcc, fps=fps, frameSize=im.size)

        for i in range(0, 1200):
            p = i
            if osp.exists(result_image_path + '/' + self.video_name.split('.')[0] + '_' + str(p) + '.jpg'):  # 判断图片是否存在
                img = cv2.imread(result_image_path + '/' + self.video_name.split('.')[0] + '_' + str(p) + '.jpg')
                # cv2.waitKey(100)
                video_writer.write(img)
                print(str(p) + '.jpg' + ' done!')

        video_writer.release()

videoimgtrans = VideoImgTrans()

videoimgtrans.img2video()