import os
import sys
from config import PATH_TO_WEIGHTS, SSD_PATH
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

sys.path.append(SSD_PATH)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd

def init_det(frame, weights=PATH_TO_WEIGHTS):
    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_weights(weights)

    image = frame#cv2.imread(frame, cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
    from matplotlib import pyplot as plt
    # from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
    # here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
    # testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
    img_id = 60
    # image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    # plt.figure(figsize=(10,10))
    # plt.imshow(rgb_image)
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    #splt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    from data import VOC_CLASSES as labels
    top_k=10

    #plt.figure(figsize=(10,10))
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    #plt.imshow(rgb_image)  # plot the image for matplotlib
    # currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    big_coords = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            if (label_name == "person"):
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = [int(pt[0]), int(pt[1]), int(pt[2]-pt[0]+1), int(pt[3]-pt[1]+1)]
                big_coords.append(coords)
                # print (coords)
                # color = colors[i]
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
    # plt.show()
    return big_coords

"""
USAGE:
init_det('frame.jpg', 'path-to-weights.pth' [OPTIONAL])
"""

