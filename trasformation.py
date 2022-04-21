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
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 58.395, 58.395]],
                reorder_channel='0 1 2') # RGB


    # Load pt model
    print('--> Loading model')
    ret = rknn.load_pytorch(model='./best.pt', input_size_list=[[3,640,640]])
    if ret != 0:
        print('Load pt failed!')
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