import torch

def IoU(input,target,threshold = 0.5):
    temp = input.clone()
    temp[temp >= threshold] = 1
    temp[temp < threshold] = 0
    #vis.image(torch.nn.functional.interpolate(temp.unsqueeze(0),size = (200, 400)))

    up = torch.sum(temp * target)
    down = torch.sum(temp) + torch.sum(target) - up
    return up / down

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

def anchor_labeling(anchor,gt,threshold = 0.4):
    batch = anchor.shape[0]
    bbox_size = anchor.shape[1]
    multi_gt = gt.expand(bbox_size,batch,4).transpose(0,1)
    keep = torch.zeros((batch,bbox_size))
    value = torch.zeros((batch,bbox_size))
    for bat in range(batch):
        temp = IoU(anchor[bat,:],multi_gt[bat,:])
        value[bat,...] = temp

        keep[bat, temp>threshold] = 1
        keep[bat, temp<=threshold] = 0

    return keep,value

def anchor_trans(anchor, bbox):
    p5_bbox = bbox
    p5_anchor = anchor
    batch = p5_bbox.shape[0]
    num_p5 = p5_anchor.shape[0]
    p5_anchor = p5_anchor.squeeze(0).expand(batch, num_p5, 4)
    p5_box = clip_boxes(bbox_transform_inv(p5_anchor, p5_bbox, batch),1,batch)
    p5_anchor = clip_boxes(p5_anchor,1,batch)

    return p5_anchor, p5_box
