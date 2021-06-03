import os.path as osp
import os
import json as js
import numpy as np
from shutil import copyfile
from PIL import Image

my_labels = ['person', 'rock', 'broom']

class DataChange:

    def __init__(self, rootpath, source_json_path, source_img_path, use_my_labels=False, need_to_change_image_size=False, image_width=300, image_height=300):
        self.rootpath = rootpath
        self.source_json_path = source_json_path
        self.source_img_path = source_img_path
        self.use_my_labels = use_my_labels
        self.need_to_change_image_size = need_to_change_image_size
        self.image_width = image_width
        self.image_height = image_height

        self.data_change()

    def produceImage(self, file_in, file_out):
        image = Image.open(file_in)
        resized_image = image.resize((self.image_width, self.image_height), Image.ANTIALIAS)
        resized_image.save(file_out)

    def data_change(self):
        xmlpath = self.rootpath + '/Annotations'
        imgpath = self.rootpath + '/JPEGImages'
        imgsetpath = self.rootpath + '/ImageSets'
        txtpath = imgsetpath + '/Main'

        if not osp.exists(xmlpath):
            os.mkdir(xmlpath)
        if not osp.exists(imgpath):
            os.mkdir(imgpath)
        if not osp.exists(imgsetpath):
            os.mkdir(imgsetpath)
        if not osp.exists(txtpath):
            os.mkdir(txtpath)

        json_file_arr = os.listdir(self.source_json_path)
        img_file_arr = os.listdir(self.source_img_path)

        json_arr = []
        img_arr = []

        for name in json_file_arr:
            if '.json' in name:
                json_arr.append(name)
        for name in img_file_arr:
            if '.jpg' in name or '.png' in name:
                img_arr.append(name)

        fixed_file_arr = []
        fixed_file_type = []

        for json in json_arr:
            for img in img_arr:
                if json.split('.')[0] == img.split('.')[0]:
                    fixed_file_arr.append(json.split('.')[0])
                    fixed_file_type.append('.' + img.split('.')[1])

        annotation_arr = np.array([])

        for i in range(len(fixed_file_arr)):

            if self.need_to_change_image_size:
                self.produceImage(self.source_img_path + '/' + fixed_file_arr[i] + fixed_file_type[i],
                                  imgpath + '/' + fixed_file_arr[i] + fixed_file_type[i])
            else:
                copyfile(self.source_img_path + '/' + fixed_file_arr[i] + fixed_file_type[i],
                         imgpath + '/' + fixed_file_arr[i] + fixed_file_type[i])

            f = open(self.source_json_path + '/' + fixed_file_arr[i] + '.json', 'r', encoding='utf-8')
            my_dic = js.load(f)
            annotation_arr = np.append(annotation_arr, (fixed_file_arr[i], my_dic))
            f.close()

        annotation_arr = annotation_arr.reshape(-1, 2)

        f1 = open(txtpath + '/test.txt', 'w')
        f2 = open(txtpath + '/trainval.txt', 'w')
        # f3 = open(txtpath + '/person_trainval.txt', 'w')
        f4 = open(txtpath + '/train.txt', 'w')
        f5 = open(txtpath + '/val.txt', 'w')
        f6 = open(self.rootpath + '/ground_truth.txt', 'w')

        for i in range(annotation_arr.shape[0]):
            f1.write(annotation_arr[i][0] + '\n')
            f2.write(annotation_arr[i][0] + '\n')
            # f3.write(annotation_arr[i][0] + ' 1\n')
            f4.write(annotation_arr[i][0] + '\n')
            f5.write(annotation_arr[i][0] + '\n')
            f6.write('\nGROUND TRUTH FOR: ' + annotation_arr[i][0] + '\n')
            f = open(xmlpath + '/' + annotation_arr[i][0] + '.xml', 'w')
            f.write('<annotation>\n')
            f.write('\t<folder>VOC2007</folder>\n')
            f.write('\t<filename>' + annotation_arr[i][0] + '</filename>\n')
            f.write('\t<size>\n')
            if self.need_to_change_image_size:
                f.write('\t\t<width>%s</width>\n' % self.image_width)
                f.write('\t\t<height>%s</height>\n' % self.image_height)
            else:
                f.write('\t\t<width>%s</width>\n' % annotation_arr[i][1]['Area']['shape'][0])
                f.write('\t\t<height>%s</height>\n' % annotation_arr[i][1]['Area']['shape'][1])
            f.write('\t\t<depth>3</depth>\n')
            f.write('\t</size>\n')
            f.write('\t<segmented>0</segmented>\n')
            if len(annotation_arr[i][1]['Area']['labels']) > 0:
                for j in range(len(annotation_arr[i][1]['Area']['labels'])):
                    f6.write('label: ')
                    f.write('\t<object>\n')
                    if self.use_my_labels:
                        f.write('\t\t<name>%s</name>\n' % my_labels[int(annotation_arr[i][1]['Area']['labels'][j][0])])
                    else:
                        f.write('\t\t<name>person</name>\n')
                    f.write('\t\t<pose>Unspecified</pose>\n')
                    f.write('\t\t<truncated>0</truncated>\n')
                    f.write('\t\t<difficult>0</difficult>\n')
                    f.write('\t\t<bndbox>\n')
                    if self.need_to_change_image_size:
                        f6.write('%d' % int(annotation_arr[i][1]['Area']['polygons'][j][0][0] * self.image_width /
                                     annotation_arr[i][1]['Area']['shape'][0]))
                        f6.write(' || ')
                        f6.write('%d' % int(annotation_arr[i][1]['Area']['polygons'][j][0][1] * self.image_width /
                                     annotation_arr[i][1]['Area']['shape'][1]))
                        f6.write(' || ')
                        f6.write('%d' % int(annotation_arr[i][1]['Area']['polygons'][j][2][0] * self.image_width /
                                     annotation_arr[i][1]['Area']['shape'][0]))
                        f6.write(' || ')
                        f6.write('%d' % int(annotation_arr[i][1]['Area']['polygons'][j][2][1] * self.image_width /
                                     annotation_arr[i][1]['Area']['shape'][1]))
                        f6.write(' || ')
                        f6.write(my_labels[int(annotation_arr[i][1]['Area']['labels'][j][0])])
                        f6.write('\n')
                        f.write('\t\t\t<xmin>%s</xmin>\n' % int(annotation_arr[i][1]['Area']['polygons'][j][0][0] * self.image_width / annotation_arr[i][1]['Area']['shape'][0]))
                        f.write('\t\t\t<ymin>%s</ymin>\n' % int(annotation_arr[i][1]['Area']['polygons'][j][0][1] * self.image_height / annotation_arr[i][1]['Area']['shape'][1]))
                        f.write('\t\t\t<xmax>%s</xmax>\n' % int(annotation_arr[i][1]['Area']['polygons'][j][2][0] * self.image_width / annotation_arr[i][1]['Area']['shape'][0]))
                        f.write('\t\t\t<ymax>%s</ymax>\n' % int(annotation_arr[i][1]['Area']['polygons'][j][2][1] * self.image_height / annotation_arr[i][1]['Area']['shape'][1]))
                    else:
                        f6.write('%d' % annotation_arr[i][1]['Area']['polygons'][j][0][0])
                        f6.write(' || ')
                        f6.write('%d' % annotation_arr[i][1]['Area']['polygons'][j][0][1])
                        f6.write(' || ')
                        f6.write('%d' % annotation_arr[i][1]['Area']['polygons'][j][2][0])
                        f6.write(' || ')
                        f6.write('%d' % annotation_arr[i][1]['Area']['polygons'][j][2][1])
                        f6.write(' || ')
                        f6.write('person\n')
                        f.write('\t\t\t<xmin>%s</xmin>\n' % annotation_arr[i][1]['Area']['polygons'][j][0][0])
                        f.write('\t\t\t<ymin>%s</ymin>\n' % annotation_arr[i][1]['Area']['polygons'][j][0][1])
                        f.write('\t\t\t<xmax>%s</xmax>\n' % annotation_arr[i][1]['Area']['polygons'][j][2][0])
                        f.write('\t\t\t<ymax>%s</ymax>\n' % annotation_arr[i][1]['Area']['polygons'][j][2][1])
                    f.write('\t\t</bndbox>\n')
                    f.write('\t</object>\n')
            f.write('</annotation>')
            f.close()
        f1.close()
        f2.close()
        # f3.close()
        f4.close()
        f5.close()
        f6.close()