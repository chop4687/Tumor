import torch
from utils.etc import IoU, clip_boxes, bbox_transform_inv, xy_to_wh,IoU2
#from RPN.utils.etc import IoU, clip_boxes, bbox_transform_inv, xy_to_wh,IoU2
def anchor_labeling(anchor,gt,threshold = 0.4):
    batch = anchor.shape[0]
    bbox_size = anchor.shape[1]
    multi_gt = gt.expand(bbox_size,batch,4).transpose(0,1)
    keep = torch.zeros((batch,bbox_size))
    value = torch.zeros((batch,bbox_size))
    for bat in range(batch):
        temp = IoU(anchor[bat,:],multi_gt[bat,:])
        value[bat,...] = temp
        #keep = value.clone()
        keep[bat, temp>threshold] = 1
        keep[bat, temp<=threshold] = 0

    return keep,value
    #
# def anchor_labeling2(anchor,gt,threshold = 0.5):
#     batch = anchor.shape[0]
#     bbox_size = anchor.shape[1]
#     keep = torch.zeros((batch,bbox_size))
#     value = torch.zeros((batch,bbox_size))
#     for bat in range(batch):
#         for bbox in range(bbox_size):
#             temp = IoU2(anchor[bat,bbox,:],gt[bat,:])
#             value[bat,bbox] = temp
#             if temp >= threshold:
#                 keep[bat,bbox] = 1
#             else:
#                 keep[bat,bbox] = 0
#
#     return keep,value

def anchor_trans(anchor, bbox, bbox_label):
    p5_bbox = bbox
    p5_anchor = anchor
    batch = p5_bbox.shape[0]
    num_p5 = p5_anchor.shape[0]
    p5_anchor = p5_anchor.squeeze(0).expand(batch, num_p5, 4)
    p5_box = clip_boxes(bbox_transform_inv(p5_anchor, p5_bbox, batch),1,batch)
    p5_anchor = clip_boxes(p5_anchor,1,batch)
    label = xy_to_wh(bbox_label)
    result = anchor_labeling(p5_anchor,bbox_label)
    return result, p5_anchor, p5_box, label
