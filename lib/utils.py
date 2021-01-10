import torch
import numpy


def calc_iou(box1, box2):
    """
    Param: box1, box2
    Return: Intersection over Union of two boxes
    
    Each boxes should be like [x1, y1, x2, y2],
    and x1 <= x2, y1 <= y2
    """
    
    (ax1, ay1, ax2, ay2) = box1
    (bx1, by1, bx2, by2) = box2
    
    assert (ax1 <= ax2) & (ay1 <= ay2)
    assert (bx1 <= bx2) & (by1 <= by2)
    
    cx1 = max(ax1, bx1)
    cy1 = max(ay1, by1)
    cx2 = min(ax2, bx2)
    cy2 = min(ay2, by2)
    
    assert (cx1 <= cx2) & (cy1 <= cy2)
        
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    c_area = (cx2 - cx1) * (cy2 - cy1)
        
    union_area = a_area + b_area - c_area
    intersecion_area = c_area
    
    smooth = 1e-6
    return (intersecion_area + smooth) / (union_area + smooth)


def calc_iou_many_to_one(boxes, ground_truth):
    """
    Param: boxes: shape([N, 4]), ground_truth: shape([4])
    Return: IoU of boxes over on ground truth box
    
    Each boxes should be like [x1, y1, x2, y2],
    and x1 <= x2, y1 <= y2
    """
    
    (gt_x1, gt_y1, gt_x2, gt_y2) = ground_truth
    boxes_x1s = boxes[:, 0]
    boxes_y1s = boxes[:, 1]
    boxes_x2s = boxes[:, 2]
    boxes_y2s = boxes[:, 3]
    
    assert (gt_x1 <= gt_x2) & (gt_y1 <= gt_y2)
    assert (boxes_x1s <= boxes_x2s).all() & (boxes_y1s <= boxes_y2s).all()
    
    inter_x1s = torch.max(boxes_x1s, gt_x1)
    inter_y1s = torch.max(boxes_y1s, gt_y1)
    inter_x2s = torch.min(boxes_x2s, gt_x2)
    inter_y2s = torch.min(boxes_y2s, gt_y2)
    
    assert (inter_x1s <= inter_x2s).all() & (inter_y1s <= inter_y2s).all()
        
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    box_areas = (boxes_x2s - boxes_x1s) * (boxes_y2s - boxes_y1s)
    intersect_areas = (inter_x2s - inter_x1s) * (inter_y2s - inter_y1s)
    
    union_area = gt_area + box_areas - intersect_areas
    intersecion_area = intersect_areas

    smooth = 1e-6    
    return (intersecion_area + smooth) / (union_area + smooth)


def determine_anchor_label(anchors, ground_truth, pos_threshold=0.7, neg_threshold=0.3):
    """
    Determine a label of anchors.
    
    Params:
        Anchors: array of [x1, y1, x2, y2]. shape([N, 4])
        ground_truth: ground truth bbox. shape([4])
        pos_threshold: IoU Threshold used to determine positive anchor
        neg_threshold: IoU Threshold used to determine negative anchor
    
    Return:
        Tensor of integer values denoting the label of anchors. shape([N])
        
        Positive: 1
        Negative: 0
        Neither positive or negative: -1
    """
    
    num_of_anchors = anchors.shape[0]
    labels = -torch.ones(num_of_anchors)
    
    ious = calc_iou_many_to_one(anchors, ground_truth)
    
    # First positive condition: Highest IoU with ground truth
    max_index = torch.argmax(ious).item()
    labels[max_index] = 1
    
    # Second positive condition: Higher than pos_threshold or equal wihh pos_threshold IoU with ground truth
    positive_flags = torch.ge(ious, pos_threshold)
    labels[positive_flags] = 1
    
    # Negative condition: Among non-positive anchors, less than neg_threshold IoU
    negative_flags = torch.eq(labels, -1) & torch.lt(ious, neg_threshold)
    labels[negative_flags] = 0
    
    return labels


# Test
if __name__ == '__main__':
    threshold = 0.0001
    
    box  = torch.tensor([2.0, 2.0, 5.0, 5.0])

    # correct answer is 0.08333333333
    box1 = torch.tensor([1.0, 1.0, 3.0, 3.0])
    assert abs(calc_iou(box, box1).item() - (1/12)) < threshold
    
    # correct answer is 0.08333333333
    box2 = torch.tensor([1.0, 4.0, 3.0, 6.0])
    assert abs(calc_iou(box, box2).item() - (1/12)) < threshold
    
    # correct answer is 1/12
    box3 = torch.tensor([4.0, 4.0, 6.0, 6.0])
    assert abs(calc_iou(box, box3).item() - (1/12)) < threshold
    
    # correct answer is 4/9
    box4 = torch.tensor([2.0, 2.0, 4.0, 4.0])
    assert abs(calc_iou(box, box4).item() - (4/9)) < threshold
    
    # correct answer is 1/9
    box5 = torch.tensor([3.0, 3.0, 4.0, 4.0])
    assert abs(calc_iou(box, box5).item() - (1/9)) < threshold
    
    ground_truth = torch.tensor([2.0, 2.0, 5.0, 5.0])
    many_boxes = torch.tensor([
        [1.0, 1.0, 3.0, 3.0],
        [1.0, 4.0, 3.0, 6.0],
        [4.0, 4.0, 6.0, 6.0],
        [2.0, 2.0, 4.0, 4.0],
        [3.0, 3.0, 4.0, 4.0]
    ])
    
    ret = calc_iou_many_to_one(many_boxes, ground_truth)
    assert torch.all(torch.lt(torch.abs(torch.add(ret, -torch.tensor([1/12, 1/12, 1/12, 4/9, 1/9]))), threshold))