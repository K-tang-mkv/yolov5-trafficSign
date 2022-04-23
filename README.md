<div align="center">

# yolov5-trafficSign </div>
Use [yolov5](https://github.com/ultralytics/yolov5) 🚀  from offical implementation as the detection algorithm to detect traffic signs including six objects, 'red light', 'green', 'yellow', 'turn_right', 'turn_left' and 'stop'.

The model is finally converted to [RKNN](https://github.com/rockchip-linux/rknn-toolkit) model that is depolyed at edge terminals to detect or run in the simulator on PC.

## <div align="center">Runtime environment</div>

* os - Ubuntu18.04 (x86_64)
* python == 3.6
* torch == 1.9.0
* tensorflow == 1.11.0
* [rknn == 1.7](https://github.com/rockchip-linux/rknn-toolkit/releases)

## <summary>Onxx2Rknn</summary>
Specific the onnx model converted to rknn in trasformation.py
```bash
python trasformation.py 
```
## <summary>Test</summary>
Test detecting one image using the rknn model, and the result is in the data/detect
```bash
python test.py
```

## <summary>Detect</summary>
It will detect the video and put the results into results.txt and data/detect/
```bash
python detect.py
