import cv2
import os
from rknn.api import RKNN

from util.letterbox import letterbox


RKNN_MODEL = 'best.rknn'
IMG_PATH = 'data/traffic/images/000002.jpg'

IMG_SIZE = 640


if __name__ == "__main__":
    print("shit")
    rknn = RKNN()

    if not os.path.exists(RKNN_MODEL):
        print('model not exist')
        exit(-1)


    # load model
    print('--> Loading model')
    ret = rknn.load_rknn(RKNN_MODEL) # success: return 0
    if ret != 0:
        print("loading model failed")
        exit(ret)
    print('done')


    # Initial runtime env
    print("--> Initial rknn runtime env")
    ret = rknn.init_runtime() #success: 0
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
    print(outputs[0].shape)
