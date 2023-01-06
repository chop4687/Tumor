import numpy as np
import torch
from torch import nn
from torchvision import transforms as T

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


def IoU2(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = torch.max(boxA[0], boxB[0])
	yA = torch.max(boxA[1], boxB[1])
	xB = torch.min(boxA[2], boxB[2])
	yB = torch.min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = torch.max(torch.zeros(1), xB - xA + 1) * torch.max(torch.zeros(1), yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def find_box_binary(img):
    img = T.ToTensor()(img)
    temp = torch.where(img>0)
    if len(temp[0]) == 0:
        x_min, y_min, x_max, y_max = 0, 0, 0, 0
    else:
        x_min = torch.min(temp[1])
        x_max = torch.max(temp[1])
        y_min = torch.min(temp[2])
        y_max = torch.max(temp[2])

    return x_min,y_min,x_max,y_max

def find_box(img):
    img = T.ToTensor()(img)
    temp = torch.where(img<1)
    x_min = torch.min(temp[1])
    x_max = torch.max(temp[1])
    y_min = torch.min(temp[2])
    y_max = torch.max(temp[2])
    return x_min, y_min, x_max, y_max

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

def xy_to_wh(bbox):
    result = torch.zeros_like(bbox)
    x1 = bbox[...,0]
    y1 = bbox[...,1]
    x2 = bbox[...,2]
    y2 = bbox[...,3]

    result[...,0] = x1
    result[...,1] = y1
    result[...,2] = x2 - x1
    result[...,3] = y2 - y1
    return result

def bbox_transform_inv(boxes, deltas, batch_size):
    # boxes = anchor 1,9,4 shape
    # 원본 deltas는 batch, roinum, 4
    # deltas = bbox 1,3,4,11,11 shape
    # center_x 임 ctr은
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dx2 = deltas[:, :, 2::4]
    dy2 = deltas[:, :, 3::4]
    dw = dx2 - dx
    dh = dy2 - dy

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape, batch_size):
    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, 374)
        boxes[i,:,1::4].clamp_(0, 374)
        boxes[i,:,2::4].clamp_(0, 374)
        boxes[i,:,3::4].clamp_(0, 374)

    return boxes
