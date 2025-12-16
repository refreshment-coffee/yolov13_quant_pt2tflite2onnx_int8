import os
import random

import onnxruntime
import numpy as np
import cv2
import torch
import torchvision
from torchvision.ops import nms

from utils import YOLOV8AnchorGenerator
import cv2
import numpy as np


##前处理---preprocess
def preprocess_params():
    input_size=1280
    keep_ratio = True
    img_norm_cfg = dict(
        mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)
    pipeline = [
        dict(type='Resize',
             size=(input_size, input_size),  # h,w
             keep_ratio=keep_ratio,
             mode='normal'),
        dict(type='LetterBox',
             size=(input_size, input_size),  # h,w
             border_value=(114, 114, 114)),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Transpose', keys=['img'], order=(2, 0, 1)),
        dict(type='Newaxis', keys=['img']),
        dict(type='Collect', keys=['img']),
    ]
    return pipeline



def preprocess_image(img, input_size=1280, keep_ratio=True):
    """
    输入：img = BGR np.ndarray, shape (H,W,3)
    输出：
        img_tensor = (1,3,640,640) float32
        meta       = dict(...)
    """

    ori_h, ori_w = img.shape[:2]
    # print(ori_h,ori_w)
    # ori_w=1280
    target_h, target_w = input_size, input_size

    # keep_ratio=False
    # =============================
    # 1) Resize (keep_ratio or not)
    # =============================
    if keep_ratio:
        r = min(target_w / ori_w, target_h / ori_h)
        # print(r)
        new_w = int(round(ori_w * r))
        new_h = int(round(ori_h * r))
        # print(new_h,new_w)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        new_h, new_w = target_h, target_w
        r = None   # 因为不保持比例

    # =============================
    # 2) LetterBox padding
    # =============================
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_pad = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    pad_shape = img_pad.shape

    # =============================
    # 3) Normalize
    # mean=[0,0,0], std=[255,255,255], to_rgb=True
    # =============================
    img_pad = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
    img_norm = img_pad.astype(np.float32) / 255.0

    # =============================
    # 4) Transpose => (3,H,W)
    # =============================
    img_chw = img_norm.transpose(2, 0, 1)

    # =============================
    # 5) Add batch dim => (1,3,H,W)
    # =============================
    img_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32)

    # =============================
    # 6) 构建 meta
    # =============================
    meta = dict(
        img_shape=(new_h, new_w, 3),
        pad_shape=pad_shape,
        scale_factor=np.array([r, r, r, r], dtype=np.float32) if keep_ratio else np.array([1, 1, 1, 1]),
        img_norm_cfg=dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True),
        pads=(top, left)  # 逆映射需要
    )

    return img_tensor, meta

def get_img_meta():
    img_meta = {
        'img_shape': (720, 1280, 3),  # 原始图像尺寸 (H, W, C)
        'pad_shape': (1280, 1280, 3),  # 填充后的图像尺寸 (H, W, C)，保持宽高比或满足模型输入
        'scale_factor': np.array([0.66667, 0.66667, 0.66667, 0.66667], dtype=np.float32),  # 缩放比例，用于将模型输出 bbox 映射回原图
        'img_norm_cfg': {
            'mean': np.array([0., 0., 0.], dtype=np.float32),  # 图像归一化的均值
            'std': np.array([255., 255., 255.], dtype=np.float32),  # 图像归一化的标准差
            'to_rgb': True  # 是否转换为 RGB
        }
    }
    return img_meta
def infer_engine(onnx_path,img_path,input_size=1280):
    img = cv2.imread(img_path)
    img_input, meta = preprocess_image(img, input_size=input_size)

    session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input.astype(np.float32)})
    return outputs,meta



def yolo_nms(bboxes, conf, scores, cfg):
    # conf_thres = cfg.score_thr
    # nms_thres = cfg.nms.iou_thr
    # max_per_img = cfg.max_per_img
    # min_bbox_size = cfg.min_bbox_size
    conf_thres = cfg['score_thr']
    nms_thres = cfg['nms']['iou_thr']
    max_per_img = cfg['max_per_img']
    min_bbox_size = cfg['min_bbox_size']

    max_wh = 4096
    wh = bboxes[:, 2:4] - bboxes[:, :2]
    print(wh)
    print(conf)
    print(max(conf))
    inds = (conf > conf_thres) & \
           ((wh > min_bbox_size) & (wh < max_wh)).all(1)
    bboxes = bboxes[inds]
    scores = scores[inds]
    print("sum(inds): ", sum(inds))
    if not bboxes.shape[0]:
        return None

    valid_inds, cls_ids = (scores > conf_thres).nonzero(as_tuple=False).t()
    det_bboxes = torch.cat(
        (bboxes[valid_inds],
         scores[valid_inds, cls_ids].unsqueeze(1),
         cls_ids.float().unsqueeze(1)), dim=1)

    inds = torch.isfinite(det_bboxes).all(1)
    det_bboxes = det_bboxes[inds]

    if not det_bboxes.shape[0]:
        return None

    boxes = det_bboxes[:, :4].clone() + \
            det_bboxes[:, 5].view(-1, 1) * max_wh
    scores = det_bboxes[:, 4]

    inds = torchvision.ops.nms(boxes, scores, nms_thres)

    if inds.shape[0] > max_per_img:
        inds = inds[: max_per_img]

    return det_bboxes[inds]

def yolo_preds(out, anchors, level_index):
    reg_max=16
    project = torch.arange(reg_max, dtype=torch.float)
    anchor_strides = [8, 16, 32]

    pred_distri, cls_scores = torch.split(
        out, [reg_max * 4, out.shape[-1] - reg_max * 4], dim=-1)  # noqa

    bbox_deltas = pred_distri.view(-1, 4, reg_max).softmax(-1)
    bbox_deltas = bbox_deltas.matmul(project.to(pred_distri.device))
    bbox_deltas = bbox_deltas.view(anchors.shape)
    bbox_deltas.mul_(bbox_deltas.new_tensor([-1, -1, 1, 1]))

    stride = anchor_strides[level_index]
    bboxes = bbox_deltas * stride + anchors

    scores = torch.sigmoid(cls_scores)

    bboxes = bboxes.view(-1, bboxes.size()[-1])
    scores = scores.view(-1, scores.size()[-1])
    conf = scores.amax(1)

    return bboxes, conf, scores


def rescale_func(dets, img_meta):
    bboxes = dets[:, :5]
    labels = dets[:, 5]
    img_shape = img_meta['img_shape']
    pad_shape = img_meta['pad_shape']
    scale_factor = img_meta['scale_factor']

    pad_left = (pad_shape[1] - img_shape[1]) * 0.5
    pad_top = (pad_shape[0] - img_shape[0]) * 0.5
    bboxes[:, [0, 2]] -= pad_left
    bboxes[:, [1, 3]] -= pad_top
    bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
    bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
    bboxes[:, :4] /= bboxes[:, :4].new_tensor(scale_factor)

    return bboxes, labels


def get_process_params():
    # return dict(
    #     nms=dict(type='nms', iou_thr=0.65),
    #     min_bbox_size=0,
    #     max_per_img=300)
    return dict(
                nms=dict(type='nms', iou_thr=0.65),
                min_bbox_size=0,
                score_thr=0.3,
                max_per_img=300
            )

def get_bboxes_single(
                      outputs,
                      mlvl_anchors,
                      img_meta):
    assert len(outputs) == len(mlvl_anchors)

    process_params=get_process_params()
    print("process_params :", process_params)
    mlvl_bboxes = []
    mlvl_conf = []
    mlvl_scores = []
    for i, x in enumerate(outputs):
        h, w = x.size()[-2:]

        # if hasattr(self, 'anchors'):
        #     x = x.view(len(self.anchors[i]), -1, h, w).permute(0, 2, 3, 1).contiguous()  # noqa
        # else:

        x = x.permute(0, 2, 3, 1).contiguous()  # noqa
        anchors = mlvl_anchors[i].to(x.device)
        bboxes, conf, scores = yolo_preds(x, anchors, i)

        mlvl_bboxes.append(bboxes)
        mlvl_conf.append(conf)
        mlvl_scores.append(scores)

    mlvl_bboxes = torch.cat(mlvl_bboxes, dim=0)
    mlvl_conf = torch.cat(mlvl_conf, dim=0)
    mlvl_scores = torch.cat(mlvl_scores, dim=0)

    # print(mlvl_conf.shape)    ##1280---torch.Size([33600])
    # print(type(mlvl_conf))
    # print(mlvl_bboxes.shape)  ##1280---torch.Size([33600],4)  cx,cy,w,h
    # print(mlvl_scores.shape)  ##1280,cls10---torch.Size([33600],10)
    # exit()
    dets = yolo_nms(mlvl_bboxes, mlvl_conf,
                         mlvl_scores, process_params)

    if dets is not None:
        bboxes, labels = rescale_func(dets, img_meta)
        return bboxes, labels
    else:
        return torch.Tensor([]), torch.Tensor([])

def get_mlvl_anchors(output, num_levels):
    anchor_strides = [8, 16, 32]
    anchor_generator = YOLOV8AnchorGenerator()
    mlvl_anchors = [
        anchor_generator.grid_anchors(
            output[i].size()[-2:], anchor_strides[i])
        for i in range(num_levels)
    ]
    return mlvl_anchors


def get_bboxes(head_outs, img_meta):
    assert head_outs[0].shape[0] == 1
    num_levels = len(head_outs)
    # print(num_levels)
    mlvl_anchors = get_mlvl_anchors(head_outs, num_levels)
    # print("mlvl_anchors: ",mlvl_anchors)
    # print(head_outs)
    # exit()
    result_list = []
    single_result = get_bboxes_single(head_outs,
                                           mlvl_anchors,
                                           img_meta)
    result_list.append(single_result)

    return result_list

def yolov8_postprocessor(outputs, img_meta):
    if outputs[0].ndim not in [3, 4]:
        raise ValueError(
            'Error Output "{}"'.format(outputs[0].ndim.shape))


    # bbox encoding
    det_results = get_bboxes(outputs, img_meta)

    return det_results


LABELS =['cls1','cls2',"cls3"]
NUM_CLASSES=len(LABELS)
COLORS = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(NUM_CLASSES)]

def visualize_boxes(img, boxes, labels):
    """
    img: np.array, BGR格式
    boxes: torch.Tensor [N,5] -> x1,y1,x2,y2,score
    labels: torch.Tensor [N] -> label_id
    """
    boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    ratio=1
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, score = box
        x1=x1*ratio
        y1 = y1 * ratio

        x2 = x2 * ratio
        y2 = y2 * ratio

        label_id = int(labels[i])
        color = COLORS[label_id % len(COLORS)]
        label_name = LABELS[label_id]

        # 画矩形框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 显示类别和分数
        text = f"{label_name} {score:.2f}"
        cv2.putText(img, text, (int(x1), max(int(y1)-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return img

"""
yolov13的Onnx变为三分支输出后，会发现，其后处理和yolov8三分支后处理是一样的。
"""
if __name__ == "__main__":
    img_path = r"/other_test_img_qiping\qiping_test5.jpg"
    onnx_path = r"D:\SYZ_projects\Project_dmx\onnx_models\best14_t3.onnx"
    # LABELS = ["class0","class1"]
    img_root=r"D:\SYZ_projects\Project_dmx\test_imgs\other_test_img_spark\test_imgs"

    data="1208_2fhmz"
    model_type="yolov13_onnx"
    save_path = os.path.join(img_root, "..",f"result_{data}_{model_type}")
    os.makedirs(save_path,exist_ok=True)
    for img_f in os.listdir(img_root):
        img_path=os.path.join(img_root,img_f)

        outputs,meta =infer_engine(onnx_path,img_path,input_size=640)
        for o in outputs:
            print(o.shape)
        # print(outputs[0][0][0][0])
        # exit()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        engine_outputs = [torch.tensor(output, device=device)
                          for output in outputs]

        # print(len(engine_outputs))
        img_meta=meta
        print("meta: ",meta)
        results = yolov8_postprocessor(engine_outputs, img_meta)[0] ##单图片预测，只取第一帧结果

        # for o in outputs:
        #     print(np.array(o).shape)
        # print(results)
        bboxs,preds=results

        img=cv2.imread(img_path)
        vis_img = visualize_boxes(img, bboxs, preds)
        # cv2.imshow("Vis_tflite", vis_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        save_img_path=os.path.join(save_path,img_f)

        cv2.imwrite(save_img_path,vis_img)