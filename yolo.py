import colorsys
import time
import os

import numpy as np
import paddle
from PIL import ImageDraw, ImageFont

from model.nets.yolo import YoloBody
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression

class YOLO(object):
    _defaults = {
        #模型文件和分类文件
        "model_path" : "weights/yolox_x.pdparams",
        "classes_path" : "model/voc_classes.txt",

        #shape大小
        "input_shape" : [640, 640],

        #使用的YoloX版本:s、m、l、x
        "phi" : "x",

        #置信度
        "confidence" : 0.5,

        #非极大抑制所用到nms_iou大小
        "nms_iou" : 0.3,

        #控制是否使用letterbox_image对图像进行不失真的resize
        "letterbox_image" : False,

        #是否使用Cuda
        "cuda" : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #初始化YOLO
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #获得种类和先验框数量
        self.class_names, self.num_classes = get_classes(self.classes_path)

        #画框设置不同颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] *255)), self.colors))
        self.generate()

    def generate(self):
        self.net = YoloBody(self.num_classes, self.phi)
        device = paddle.device.get_device()
        self.net.load_state_dict(paddle.load(self.model_path))
        self.net = self.net.eval()

        print("{} model, and classes loaded.".format(self.model_path))

        if self.cuda:
            self.net = paddle.DataParallel(self.net)
            self.net.train()

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)

        #给图像增加灰条，实现不失真的resize
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with paddle.no_grad():
            images = paddle.to_tensor(image_data)
            if self.cuda:
                images = images.cpu()

            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)

            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with paddle.no_grad():
            images = paddle.to_tensor(image_data)
            if self.cuda:
                images = images.cpu()

        t1 = time.time()
        for _ in range(test_interval):
            with paddle.no_grad():
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                              image_shape, self.letterbox_image, conf_thres=self.confidence,
                                              nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with paddle.no_grad():
            images = paddle.to_tensor(image_data)
            if self.cuda:
                images = images.cpu()

            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)

            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in self.class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return