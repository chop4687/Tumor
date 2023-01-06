import numpy as np
import torch
from visdom import Visdom
from torchvision import transforms as T
from utils.transbox import anchor_trans
from torch import nn
import os

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def find_box(img):
    img = img.resize((3000,1500))
    img = np.array(img)
    temp = np.where(img<255)
    x_min = np.min(temp[0])
    x_max = np.max(temp[0])
    y_min = np.min(temp[1])
    y_max = np.max(temp[1])
    return x_min, y_min, x_max, y_max

def find_box_binary(img):
    img[img<1] = 0
    temp = torch.where(img>0)

    if len(temp[0]) == 0:
        x_min, y_min, x_max, y_max = 0, 0, 0, 0
    else:
        x_min = torch.min(temp[1])
        x_max = torch.max(temp[1])
        y_min = torch.min(temp[2])
        y_max = torch.max(temp[2])

    x_cen = int(x_min+(x_max-x_min)/2)
    y_cen = int(y_min+(y_max-y_min)/2)
    return x_cen, y_cen

def find_center(ten_img):
    ten_img[ten_img<1] = 0
    temp = torch.where(ten_img==1)
    if len(temp[0]) == 0:
        x_min, y_min, x_max, y_max = 0, 0, 0, 0
    else:
        x_min = torch.min(temp[1])
        x_max = torch.max(temp[1])
        y_min = torch.min(temp[2])
        y_max = torch.max(temp[2])

    x_cen = int(x_min+(x_max-x_min)/2)
    y_cen = int(y_min+(y_max-y_min)/2)
    return x_cen, y_cen

def save_txt(epoch,predict_score,predict_patch_score,class_y,name):
    fw = open('./result/'+str(epoch)+'.txt','w')
    fw.write(str(class_y))
    fw.write(str(name))
    fw.write(str(predict_score))
    fw.write(str(predict_patch_score))
    fw.close()
    return print('save_end')

def conf_matrix(input,target):
    # input B n_Class
    # target B Class
    preds = torch.argmax(input,1)

    zero_idx = (target == 0).nonzero()
    one_idx = (target == 1).nonzero()
    zz = torch.sum(preds[zero_idx] == 0)
    zo = torch.sum(preds[zero_idx] == 1)
    oz = torch.sum(preds[one_idx] == 0)
    oo = torch.sum(preds[one_idx] == 1)
    print(f'{zz},{zo}\n{oz},{oo}')
########################################################
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

def IoU(boxes1, boxes2):
    area1 = box_area(boxes1) #batch,area
    area2 = box_area(boxes2) #batch,area
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt + 1).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    iou = inter / (area1 + area2 - inter)
    return iou

def generate_anchors_single_pyramid(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # # Reshape to get a list of (y, x) and a list of (h, w)
    # box_centers = np.stack(
    #     [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    # box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # NOTE: the original order is  (y, x), we changed it to (x, y) for our code
    # Reshape to get a list of (x, y) and a list of (w, h)
    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (x1, y1, x2, y2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes


def generate_anchors_all_pyramids(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors_single_pyramid(scales[i], ratios, feature_shapes,
                                        feature_strides, anchor_stride))
    return torch.FloatTensor(np.concatenate(anchors, axis=0))


def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())

def heatmap(x, model):
    cls, bbox, temp_c2, temp_c3, temp_c4, temp_c5 = model(x)
    black = torch.zeros((x.shape[0],375,375)).to(0)
    for i in range(x.shape[0]):
        p5_anchor = generate_anchors_all_pyramids(scales=[256],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
        p4_anchor = generate_anchors_all_pyramids(scales=[128],ratios=(0.5,1,2),feature_shapes=([5,5]),feature_strides=75,anchor_stride=1).to(0)
        p3_anchor = generate_anchors_all_pyramids(scales=[64],ratios=(0.5,1,2),feature_shapes=([11,11]),feature_strides=34,anchor_stride=1).to(0)
        p2_anchor = generate_anchors_all_pyramids(scales=[32],ratios=(0.5,1,2),feature_shapes=([23,23]),feature_strides=17,anchor_stride=1).to(0)
        total_anchor = torch.cat((p2_anchor, p3_anchor, p4_anchor, p5_anchor))
        anchor, predict_anchor = anchor_trans(total_anchor, bbox[i,...].unsqueeze(0))
        arg_cls = nn.Softmax(dim=2)(cls[i,...].unsqueeze(0))
        value,idx = torch.sort(arg_cls[...,1],descending=True)
        pred = predict_anchor[0,idx[0,0:100]]
        for k in range(5):
            if arg_cls[0,idx[0,k],1] >= 0.5:
                x_min, y_min, w, h = pred[k,:].int()
                black[i, x_min:(x_min + w),y_min:(y_min+h)] += arg_cls[0,idx[0,k],1]
                #black[i,...] = scale(black[i,...])
    return cls, bbox, temp_c2, temp_c3, temp_c4, temp_c5, black
