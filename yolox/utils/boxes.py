#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision
import torch.nn.functional as F

try:
    import os
    import sys
    pwd = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(1, os.path.join(pwd, "../../Rotated_IoU"))
    from oriented_iou_loss import cal_iou
except Exception as e:
    cal_iou = None

__all__ = [
    "postprocess_rotate",
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cal_angle"
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [torch.empty((0, 7), device=prediction.device) for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]

        output[i] = torch.cat((output[i], detections))

    return output


def postprocess_rotate(size):  # (h, w)
    def postprocess_rotate(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False, size=size):
        
        # cal angle from (cxcy) to center of img
        angle = cal_angle(prediction[..., :4], size)
        prediction = torch.cat([prediction[..., :4], angle, prediction[..., 4:]], dim=-1)

        # box_corner = prediction.new(prediction.shape)
        # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        # prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 6: 6 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 5] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, angle(rad), obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :6], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            # if class_agnostic:
            #     nms_out_index = torchvision.ops.nms(
            #         detections[:, :4],
            #         detections[:, 5] * detections[:, 6],
            #         nms_thre,
            #     )
            # else:
            #     nms_out_index = torchvision.ops.batched_nms(
            #         detections[:, :4],
            #         detections[:, 5] * detections[:, 6],
            #         detections[:, 7],
            #         nms_thre,
            #     )
            nms_out_index = rotated_nms(
                detections[:, :5],
                detections[:, 5] * detections[:, 6],
                detections[:, 7],
                nms_thre
            )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output
    return postprocess_rotate

def rotated_nms(boxes, scores, cls, iou_threshold):
    """
        non maximum suppression kernel for polygon-enabled boxes
        x is the prediction with boxes x[:, :8], confidence x[:, 8], class x[:, 9] 
        Return the selected indices
    """
    
    unique_labels = cls.unique()
    _, scores_sort_index = torch.sort(scores, descending=True)
    boxes = boxes[scores_sort_index]
    indices = scores_sort_index
    selected_indices = []
    
    # Iterate through all predicted classes
    for unique_label in unique_labels:
        boxes_ = boxes[cls==unique_label]
        indices_ = indices[cls==unique_label]
        
        while boxes_.shape[0]:
            # Save the indice with the highest confidence
            selected_indices.append(indices_[0])
            if len(boxes_) == 1: break
            # Compute the IOUs for all other the polygon boxes
            iou = cal_iou(boxes_[0:1].expand_as(boxes_[1:]).reshape(1, -1, 5), boxes_[1:].reshape(1, -1, 5))[0][0]
            # Remove overlapping detections with IoU >= NMS threshold
            boxes_ = boxes_[1:][iou < iou_threshold]
            indices_ = indices_[1:][iou < iou_threshold]
            
    return torch.LongTensor(selected_indices)


def cal_angle(bboxes, size):
    '''
    input: bboxes - (..., 4) - cx,cy,w,h
    output: angle - (..., 1) - a(radian)
    '''
    b = bboxes[..., :2].detach() - torch.tensor([size[1]/2, size[0]/2], device=bboxes.device)
    a = torch.tensor([0, -10], device=bboxes.device).expand_as(b)
    inner_product = (a * b).sum(dim=-1)
    a_norm = a.pow(2).sum(dim=-1).pow(0.5)
    b_norm = b.pow(2).sum(dim=-1).pow(0.5)
    cos = inner_product / (1 * a_norm * b_norm)
    angle = torch.acos(cos)
    angle[b[..., 0] < 0] *= -1

    # a_norm = F.normalize(a, p=2, dim=-1)
    # b_norm = F.normalize(b, p=2, dim=-1)
    # inner_product = (a_norm * b_norm).sum(dim=-1)
    # angle = torch.acos(inner_product)
    # angle[b[..., 0] < 0] *= -1

    return angle[..., None]


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
