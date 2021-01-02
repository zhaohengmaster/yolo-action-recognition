from collections import deque
import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model

# from gluoncv.utils.filesystem import try_import_decord

SAMPLE_DURATION = 32
frames = deque(maxlen=SAMPLE_DURATION)


video_file = "smoking3people.mov"

# load video
vid = cv2.VideoCapture(video_file)
model_name = 'i3d_resnet50_v1_hmdb51'
net = get_model(model_name, pretrained=True)

while True:
    ret, frame = vid.read()
    if not ret:
        break
    frames.append(frame)
    # print([f.shape for f in nframes])
    
    # cv2.imshow("frame", frame)
    if len(frames) < SAMPLE_DURATION:
        continue

    # if len(clip_input) == SAMPLE_DURATION:
    clip_input = frames
    transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_input = transform_fn(clip_input)
    clip_input = np.stack(clip_input, axis=0)
    clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    pred = net(nd.array(clip_input))
    classes = net.classes
    topK = 1
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input video clip is classified to be')
    for i in range(topK):
        print('\t[%s], with probability %.3f.'%
            (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
    # break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break