import torch
import torch.nn as nn


def smooth_L1(ti, ti_star):
    """
    smooth L1 function:
        0.5 * (x^2) if abs(x) < 1
        abs(x) - 0.5 otherwise

    Params:
        ti: shape([N])
        ti_star: shape([N])
    
    Return: score: shape([N])
    """
    abs_sub = torch.abs(ti - ti_star)
    
    smaller_than_1 = torch.where(abs_sub < 1)
    greater_than_1 = torch.where(abs_sub >= 1)
    
    abs_sub[smaller_than_1] = torch.pow(abs_sub[smaller_than_1], 2) / 2
    abs_sub[greater_than_1] = abs_sub[greater_than_1] - 0.5
        
    return abs_sub


def rpn_loss_reg(pred_boxes, anchor_boxes, gt_box):
    # TODO: gt_box? or gt_boxes?
    """
    Regression loss of RPN Layer.
    
    Params:
        pred_boxes: Predicted boxes by RPN layer. shape([N, 4])
        anchor_boxes: Anchor boxes used by the predictions. shape([N, 4])
        gt_box: Ground truth box of image. shape([4])
    """
    
    x = pred_boxes[:, 0]
    y = pred_boxes[:, 1]
    w = pred_boxes[:, 2] - pred_boxes[:, 0]
    h = pred_boxes[:, 3] - pred_boxes[:, 1]

    x_a = anchor_boxes[:, 0]
    y_a = anchor_boxes[:, 1]
    w_a = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    h_a = anchor_boxes[:, 3] - anchor_boxes[:, 1]

    x_star = gt_box[0]
    y_star = gt_box[1]
    w_star = gt_box[2] - gt_box[0]
    h_star = gt_box[3] - gt_box[1]
    
    t_x = (x - x_a) / w_a
    t_y = (y - y_a) / h_a
    t_w = torch.log(w/w_a)
    t_h = torch.log(h/h_a)
    
    t_x_star = (x_star - x_a) / w_a
    t_y_star = (y_star - y_a) / h_a
    t_w_star = torch.log(w_star/w_a)
    t_h_star = torch.log(h_star/h_a)
    
    losses = torch.zeros(anchor_boxes.shape[0])
    losses += smooth_L1(t_x, t_x_star)
    losses += smooth_L1(t_y, t_y_star)
    losses += smooth_L1(t_w, t_w_star)
    losses += smooth_L1(t_h, t_h_star)
    
    return losses


def rpn_loss_cls(preds, labels):
    """
    Classification loss of RPN Layer.
    Log loss between probability that anchor is object and binary ground truth label
    
    Params:
        preds: Probabilities that anchors are objects
        labels: Labels that anchors are objects
    """
    
    assert torch.all(torch.ge(preds, 0.0))
    assert torch.all(torch.le(preds, 1.0))
    
    binary_cross_entropy = nn.BCELoss(reduction='none')
    output = binary_cross_entropy(preds, labels)
    return output


def multitask_loss(pred_probs,
                   pred_boxes, anchor_boxes, gt_box,
                   anchor_num=9, balance=10):
    """
    
    L(p, t) = (1/N_cls) * sigma{L_cls(pi, pi_star)} + lambda * (1/N_reg) * sigma{pi_star * L_reg(ti, ti_star)}
    """
    
    # Positive: 1 Negative: 0 Neither positive or negative: -1
    labels = determine_anchor_label(anchor_boxes, gt_box)
    
    # Only get positive and negative anchors
    valid_indices = torch.where(labels > -0.5)
    valid_labels = labels[valid_indices]
    valid_pred_probs = pred_probs[valid_indices]
    valid_pred_boxes = pred_boxes[valid_indices]
    valid_anchor_boxes = anchor_boxes[valid_indices]
    
    cls_loss = rpn_loss_cls(valid_pred_probs, valid_labels)
    reg_loss = rpn_loss_reg(valid_pred_boxes, valid_anchor_boxes, gt_box)
    positive_reg_loss = reg_loss * valid_labels
    
    n_cls = anchor_boxes.shape[0] / anchor_num
    n_reg = anchor_boxes.shape[0]
    
    cls_term = torch.sum(cls_loss) / n_cls
    reg_term = torch.sum(positive_reg_loss) / n_reg * balance
    
    return cls_term + reg_term
