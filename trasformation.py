'''
Convert onnx to rknn
Export RKNN model
'''

from rknn.api import RKNN

if __name__ == '__main__':
    # Create RKNN execution objects
    print("shit")
    rknn = RKNN()


    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='103.94 116.78 123.68 58.82',
                reorder_channel='0 1 2') # RGB


    # Load onnx model
    print('--> Loading model')
    ret = rknn.load_onnx(model='./onnx/best.onnx')
    if ret != 0:
        print('Load onnx failed!')
        exit(ret)
    print('done')


    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build yoloRknn failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./best.rknn')
    if ret != 0:
        print('Export best.rknn failed!')
        exit(ret)
    print('done')

    #release rknn context
    rknn.release()