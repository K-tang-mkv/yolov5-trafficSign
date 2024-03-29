'''
Convert onnx to rknn
Export RKNN model
'''

from rknn.api import RKNN

ONNX_MODEL = './onnx/best.onnx'
RKNN_MODEL = 'best.rknn'
DATASET = 'dataset.txt'

if __name__ == '__main__':
    # Create RKNN execution objects
    print("shit")
    rknn = RKNN()


    # pre-process config
    print('--> config model')
    rknn.config(
                batch_size=8,
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                reorder_channel='0 1 2',
                target_platform=None) # RGB


    # Load onnx model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['326', '371', '416'])
    if ret != 0:
        print('Load pt failed!')
        exit(ret)
    print('done')


    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    if ret != 0:
        print('Build yoloRknn failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export best.rknn failed!')
        exit(ret)
    print('done')

    #release rknn context
    rknn.release()
