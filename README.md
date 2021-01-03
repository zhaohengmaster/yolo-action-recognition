# Person Action Recognition

Classify person action with yolo, deep sort and gluoncv. With using tracking, we can
classify each person in the video

![](samples/samples.gif)

## Installation

OS X & Linux:

```sh
create -f environment.yml
```

Windows:

```
Have problem when install mxnet and gluoncv with GPU Support
```

## Usage 

```python
python run.py --source [path of video]
```

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [YoloV5](https://github.com/ultralytics/yolov5)
* [Deep Sort Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
* [pyimagesearch](https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/)
