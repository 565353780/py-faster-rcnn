import os.path as osp
import os
import cv2
from datachange import my_labels
import numpy as np
import matplotlib.pyplot as plt

img_type = '.jpg'

class DrawRectangle:

    def __init__(self, rootpath, net_name=None, input_image=None, class_name=None, dets=None, thresh=0.5, need_to_draw_rectangle=False, line_color=[0, 0, 255], line_width=5, need_to_show_name=False, name_color=(0, 255, 0), use_my_labels=False, need_to_evaluate_result=False):
        self.rootpath = rootpath
        self.net_name = net_name
        self.input_image = input_image
        self.class_name = class_name
        self.dets = dets
        self.thresh = thresh
        self.need_to_draw_rectangle = need_to_draw_rectangle
        self.line_color = line_color
        self.line_width = line_width
        self.need_to_show_name = need_to_show_name
        self.name_color = name_color
        self.result_arr = []
        self.use_my_labels = use_my_labels
        self.need_to_evaluate_result = need_to_evaluate_result
        self.ground_truth_arr = []
        self.eval_arr = []
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.Precision = 0
        self.Recall = 0

        if self.input_image is None:
            self.load_data()

            if self.need_to_draw_rectangle:
                self.draw_rectangle()

            if self.need_to_evaluate_result:
                self.evaluate_result()
                self.show_eval_msg()
        else:
            if self.need_to_draw_rectangle:
                self.draw_rectangle()

    def load_data(self):
        
        if not osp.exists(self.rootpath + '/results'):
            os.mkdir(self.rootpath + '/results')
        if self.net_name is not None:
            if not osp.exists(self.rootpath + '/results/' + self.net_name):
                os.mkdir(self.rootpath + '/results/' + self.net_name)

        total_rows = 0
        row_idx = 0

        total_ground_truth = 0
        ground_truth_idx = 0

        f = open(self.rootpath + '/result_data.txt', 'r')
        lines_f = f.readlines()

        for line in lines_f:
            if ')' in line:
                total_rows += 1
            if line.split(' ')[0] == 'label:':
                total_ground_truth += 1

        self.result_arr = [0] * total_rows
        for i in range(total_rows):
            self.result_arr[i] = [0] * 7

        temp_filename = ' '

        self.ground_truth_arr = [0] * total_ground_truth
        for i in range(total_ground_truth):
            self.ground_truth_arr[i] = [0] * 7

        for line in lines_f:
            if line is not '\n':

                if 'GROUND' in line:
                    temp_filename = line.split('\n')[0].split(': ')[1]

                if line.split(' ')[0] == 'label:':
                    self.ground_truth_arr[ground_truth_idx][0] = temp_filename
                    if self.use_my_labels:
                        self.ground_truth_arr[ground_truth_idx][1] = my_labels[int(line.split('\n')[0].split(' ')[9])]
                    else:
                        self.ground_truth_arr[ground_truth_idx][1] = 'person'
                    self.ground_truth_arr[ground_truth_idx][2] = line.split(' ')[1]
                    self.ground_truth_arr[ground_truth_idx][3] = line.split(' ')[3]
                    self.ground_truth_arr[ground_truth_idx][4] = line.split(' ')[5]
                    self.ground_truth_arr[ground_truth_idx][5] = line.split(' ')[7]
                    ground_truth_idx += 1

                if ')' in line:
                    self.result_arr[row_idx][0] = temp_filename
                    self.result_arr[row_idx][1] = line.split('\n')[0].split(': ')[1].split(' ')[0]
                    self.result_arr[row_idx][2] = line.split('\n')[0].split(') ')[1].split(' || ')[0]
                    self.result_arr[row_idx][3] = line.split('\n')[0].split(') ')[1].split(' || ')[1]
                    self.result_arr[row_idx][4] = line.split('\n')[0].split(') ')[1].split(' || ')[2]
                    self.result_arr[row_idx][5] = line.split('\n')[0].split(') ')[1].split(' || ')[3]
                    row_idx += 1

    def draw_rectangle(self):

        if self.input_image is None:

            total_rows = len(self.result_arr)

            i = 0
            while i < total_rows:
                i_end = i

                while i_end < total_rows - 1 and self.result_arr[i_end + 1][0] == self.result_arr[i_end][0]:
                    i_end += 1

                image = cv2.imread(self.rootpath + '/JPEGImages/' + self.result_arr[i][0] + img_type)

                while i < i_end + 1:
                    for j in range(int(float(self.result_arr[i][2])), int(float(self.result_arr[i][4])) + 1):
                        if -1 < j < image.shape[1]:
                            if -1 < int(float(self.result_arr[i][3])) < image.shape[0]:
                                image[int(float(self.result_arr[i][3]))][j] = self.line_color
                            for k in range(1, self.line_width + 1):
                                if -1 < int(float(self.result_arr[i][3])) - k < image.shape[0]:
                                    image[int(float(self.result_arr[i][3])) - k][j] = self.line_color
                                if -1 < int(float(self.result_arr[i][3])) + k < image.shape[0]:
                                    image[int(float(self.result_arr[i][3])) + k][j] = self.line_color
                            if -1 < int(float(self.result_arr[i][5])) < image.shape[0]:
                                image[int(float(self.result_arr[i][5]))][j] = self.line_color
                            for k in range(1, self.line_width + 1):
                                if -1 < int(float(self.result_arr[i][5])) - k < image.shape[0]:
                                    image[int(float(self.result_arr[i][5])) - k][j] = self.line_color
                                if -1 < int(float(self.result_arr[i][5])) + k < image.shape[0]:
                                    image[int(float(self.result_arr[i][5])) + k][j] = self.line_color
                    for j in range(int(float(self.result_arr[i][3])), int(float(self.result_arr[i][5])) + 1):
                        if -1 < j < image.shape[0]:
                            if -1 < int(float(self.result_arr[i][2])) < image.shape[1]:
                                image[j][int(float(self.result_arr[i][2]))] = self.line_color
                            for k in range(1, self.line_width + 1):
                                if -1 < int(float(self.result_arr[i][2])) - k < image.shape[1]:
                                    image[j][int(float(self.result_arr[i][2])) - k] = self.line_color
                                if -1 < int(float(self.result_arr[i][2])) + k < image.shape[1]:
                                    image[j][int(float(self.result_arr[i][2])) + k] = self.line_color
                            if -1 < int(float(self.result_arr[i][4])) < image.shape[1]:
                                image[j][int(float(self.result_arr[i][4]))] = self.line_color
                            for k in range(1, self.line_width + 1):
                                if -1 < int(float(self.result_arr[i][4])) - k < image.shape[1]:
                                    image[j][int(float(self.result_arr[i][4])) - k] = self.line_color
                                if -1 < int(float(self.result_arr[i][4])) + k < image.shape[1]:
                                    image[j][int(float(self.result_arr[i][4])) + k] = self.line_color

                    if self.need_to_show_name:

                        if int(float(self.result_arr[i][2])) < 0:
                            text_x = 0
                        elif int(float(self.result_arr[i][2])) > image.shape[1]:
                            text_x = image.shape[1] - 1
                        else:
                            text_x = int(float(self.result_arr[i][2]))
                        if int(float(self.result_arr[i][3])) < 0:
                            text_y = 0
                        elif int(float(self.result_arr[i][3])) > image.shape[0]:
                            text_y = image.shape[0] - 1
                        else:
                            text_y = int(float(self.result_arr[i][3]))
                        org = (text_x, text_y - 20)
                        fontFace = cv2.FONT_HERSHEY_COMPLEX
                        fontScale = 2
                        color = self.name_color
                        thickness = 2
                        lineType = 8
                        bottomLeftOrigin = False
                        cv2.putText(image, self.result_arr[i][1], org, fontFace, fontScale, color, thickness, lineType,
                                    bottomLeftOrigin)
                    i += 1

                cv2.imwrite(self.rootpath + '/results/' + self.net_name + '/' + self.result_arr[i - 1][0] + img_type, image)

                if i < total_rows:
                    print('process: %s/%s....' % (i + 1, total_rows))
                else:
                    print('process: %s/%s....' % (total_rows, total_rows))
        else:
            inds = np.where(self.dets[:, -1] >= self.thresh)[0]
            if len(inds) == 0:
                return

            for i in inds:
                bbox = self.dets[i, :4]
                score = self.dets[i, -1]

                for j in range(int(bbox[0]), int(bbox[2]) + 1):
                    self.input_image[int(bbox[1])][j] = self.line_color
                    for k in range(1, self.line_width + 1):
                        if int(bbox[1]) - k > -1:
                            self.input_image[int(bbox[1]) - k][j] = self.line_color
                        if int(bbox[1]) + k < self.input_image.shape[0]:
                            self.input_image[int(bbox[1]) + k][j] = self.line_color
                    self.input_image[int(bbox[3])][j] = self.line_color
                    for k in range(1, self.line_width + 1):
                        if int(bbox[3]) - k > -1:
                            self.input_image[int(bbox[3]) - k][j] = self.line_color
                        if int(bbox[3]) + k < self.input_image.shape[0]:
                            self.input_image[int(bbox[3]) + k][j] = self.line_color
                for j in range(int(bbox[1]), int(bbox[3]) + 1):
                    self.input_image[j][int(bbox[0])] = self.line_color
                    for k in range(1, self.line_width + 1):
                        if int(bbox[0]) - k > -1:
                            self.input_image[j][int(bbox[0]) - k] = self.line_color
                        if int(bbox[0]) + k < self.input_image.shape[1]:
                            self.input_image[j][int(bbox[0]) + k] = self.line_color
                    self.input_image[j][int(bbox[2])] = self.line_color
                    for k in range(1, self.line_width + 1):
                        if int(bbox[2]) - k > -1:
                            self.input_image[j][int(bbox[2]) - k] = self.line_color
                        if int(bbox[2]) + k < self.input_image.shape[1]:
                            self.input_image[j][int(bbox[2]) + k] = self.line_color
                if self.need_to_show_name:
                    org = (int(bbox[0]), int(bbox[1]) - 20)
                    fontFace = cv2.FONT_HERSHEY_COMPLEX
                    fontScale = 2
                    color = self.name_color
                    thickness = 2
                    lineType = 8
                    bottomLeftOrigin = False
                    cv2.putText(self.input_image, self.class_name, org, fontFace, fontScale, color, thickness, lineType,
                                bottomLeftOrigin)

    def evaluate_result(self):

        self.eval_arr = [0] * len(self.result_arr)

        result_idx = 0

        for result in self.result_arr:
            for ground_truth in self.ground_truth_arr:
                if result[0] == ground_truth[0]:
                    min_x = max(float(result[2]), float(ground_truth[2]))
                    min_y = max(float(result[3]), float(ground_truth[3]))
                    max_x = min(float(result[4]), float(ground_truth[4]))
                    max_y = min(float(result[5]), float(ground_truth[5]))
                    if min_x < max_x and min_y < max_y:
                        current_accu = (max_x - min_x) * (max_y - min_y) / (float(ground_truth[4]) - float(ground_truth[2])) / (float(ground_truth[5]) - float(ground_truth[3]))
                        self.eval_arr[result_idx] = max(self.eval_arr[result_idx], current_accu)
                        if current_accu > 0.5:
                            ground_truth[6] = 1

            if self.eval_arr[result_idx] > 0.5:
                result[6] = 1

            result_idx += 1

        for value in self.ground_truth_arr:
            if value[6] == 1:
                self.true_positive += 1
            else:
                self.false_negative += 1
        for value in self.result_arr:
            if value[6] == 0:
                self.false_positive += 1

        self.Precision = self.true_positive / (self.true_positive + self.false_positive)
        self.Recall = self.true_positive / (self.true_positive + self.false_negative)

    def show_eval_msg(self):

        temp_image = ' '
        temp_idx = 1

        f = open(self.rootpath + '/output_msg.txt', 'a+')
        f.write('-------------------------------------------\n')
        for i in range(len(self.eval_arr)):
            if self.result_arr[i][0] != temp_image:
                temp_image = self.result_arr[i][0]
                temp_idx = 1
            else:
                temp_idx += 1
            print('Image : ' + self.result_arr[i][0] + '.png' + ' , Person : %d , Accuracy : %.2f%%' % (
            temp_idx, self.eval_arr[i] * 100))
            f.write('Image : ' + self.result_arr[i][0] + '.png' + ' , Person : %d , Accuracy : %.2f%%\n' % (
            temp_idx, self.eval_arr[i] * 100))
            print('-------------------------------------------')
            f.write('-------------------------------------------\n')
        print('true positive  : %d' % self.true_positive)
        print('true negative  : %d' % self.true_negative)
        print('false positive : %d' % self.false_positive)
        print('false negative : %d' % self.false_negative)
        print('Precision      : %.2f%%' % (self.Precision * 100))
        print('Recall         : %.2f%%' % (self.Recall * 100))
        f.write('true positive  : %d\n' % self.true_positive)
        f.write('true negative  : %d\n' % self.true_negative)
        f.write('false positive : %d\n' % self.false_positive)
        f.write('false negative : %d\n' % self.false_negative)
        f.write('Precision      : %.2f%%\n' % (self.Precision * 100))
        f.write('Recall         : %.2f%%\n' % (self.Recall * 100))
        f.write('-------------------------------------------\n')


        x = np.linspace(1, len(self.eval_arr), len(self.eval_arr))
        sorted_eval_arr = sorted(self.eval_arr)
        plt.plot(x, self.eval_arr)
        plt.show()
        plt.plot(x, sorted_eval_arr)
        plt.show()
