import numpy as np
import cv2
from rknn.api import RKNN 
import os
from util import yolov5_post_process, letterbox, draw


CLASSES = ('red', 'yellow', 'green', 'turn_reight', 'turn_left', 'stop')
IMG_SIZE = 640
IMG_PATH = 'data/traffic/images/000002.jpg'
RKNN_MODEL = './best.rknn'

if __name__ == "__main__":
    print("shit")
    rknn = RKNN()

    if not os.path.exists(RKNN_MODEL):
        print('model not exist')
        exit(-1)

    # load model
    print('--> Loading model')
    ret = rknn.load_rknn(RKNN_MODEL)  # success: return 0
    if ret != 0:
        print("loading model failed")
        exit(ret)
    print('done')

    # Initial runtime env
    print("--> Initial rknn runtime env")
    ret = rknn.init_runtime()  # success: 0
    if ret != 0:
        print("Initial rknn runtime failed")
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference
    print('--> Running model to inference:')
    outputs = rknn.inference(inputs=[img])

    input0_data = outputs[0]
    input1_data = outputs[1]
    input2_data = outputs[2]

    input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = yolov5_post_process(input_data)

    
    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_1, boxes, scores, classes)
    cv2.imwrite("./data/detect/im.jpg", img_1)
    cv2.waitKeyEx(0)

