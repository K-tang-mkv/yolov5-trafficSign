import os
import time
import numpy as np
import cv2
from rknn.api import RKNN
from util import letterbox, yolov5_post_process, draw

RKNN_MODEL = 'best.rknn'
BOX_THRESH = 0.5
NMS_THRESH = 0.3
IMG_SIZE = 640
CLASSES = {'red', 'yellow', 'green', 'turn_left', 'turn_right', 'stop'}

if __name__ == '__main__':
    # create a RKNN object
    rknn = RKNN()

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


    # Handle input data
    capture = cv2.VideoCapture('./data/video.mp4') # read video
    count = 0   # record frames
    ret, img = capture.read()  # if can read one frame, then return 1 to ret and the frame to image


    # Handle results
    if os.path.exists(r'results.txt'):
        os.remove(r'results.txt') # remove results
    if os.path.exists(r'./data/detect'):
        for f in os.listdir('./data/detect'):
             os.remove(os.path.join('./data/detect', f))
    with open('results.txt', 'a') as f:      # put results into results.txt
        while(ret):
            img_1, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

            # Run the model
            print('--> Running model')
            time1 = time.time()
            outputs = rknn.inference(inputs=[img_1])
            time2 = time.time()
            print((time2-time1)*1000)
            
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
            # Handle the inferenced outputs
            boxes, classes, scores = yolov5_post_process(input_data)

            if boxes is not None:      # if there was boxes appearing in the img, then return the format saved to results.txt and draw the img with boxes
                tmp = str(count) + ', ' + str((time2-time1)*1000) + ', ' + draw(img_1, boxes, scores, classes, count) + '\n'
                # format: frame_count, delay, classes, x0, y0, x1, y1
            else:
                tmp = str(count) + ', ' + str((time2-time1)*1000) + '\n'

            f.write(tmp)
            #cv2.imwrite('./data/detect/img' + str(count) + '.jpg', img_1)

            count += 1
            ret, img = capture.read()       # read the next frame to inference


    rknn.release()






