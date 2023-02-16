import argparse
import json
import os
import sys
from pathlib import Path
import torchvision
import numpy as np
import torch
from tqdm import tqdm
import kornia
import cv2
from matplotlib import pyplot as plt
import random


FILE = Path(__file__).resolve()
ROOT = Path(os.getcwd())  # YOLOv5 root directory
if str(ROOT) not in sys.path:
   sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

def parse_opt(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args(arguments)
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    xd1  = box1.unsqueeze(1).chunk(2, 2)
    xd2 = box2.unsqueeze(0).chunk(2, 2)
    gt1, gt2 = box1.unsqueeze(1).chunk(2,2)
    p1, p2 = box2.unsqueeze(0).chunk(2,2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    hehe=torch.min(a2,b2)
    haha = torch.max(a1,b1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def bboxes_to_lines(bboxes):
    """
    bboxes - [N,4] N is amount of bboxes defined by xyxy annotation (: 
    
    returns:
    lines - [N,4] a line defined by 2 points it passes through - both are on
    the vertical edge of the given boudning box (at the half of it exactly)
    """
    bboxes = bboxes.to(torch.int32)
    lines = torch.empty((bboxes.shape[0], 4), dtype=torch.int32)
    for i, box in enumerate(bboxes):
        assert box[1] <= box[3], f"y2 must be bigger than y1 in predicted bounding boxes gotten: y1={box[1]} y2={box[3]}" #i want to be sure that y2 > y1
        bbox_height = box[3] - box[1]
        x1, y1 = box[0], box[1] + int(bbox_height/2)
        x2, y2 = box[2], box[3] - int(bbox_height/2)
        lines[i] = torch.tensor([x1, y1, x2, y2], dtype=torch.int32)
    return lines

def keypoints_to_lines(lines):
    """
    lines = [N,4] lines in x1,y1,x2,y2 format
    returns:
    keypoints = []
    """
    for line in lines:
        pass



def _draw_pixel(image: torch.Tensor, x: int, y: int, color: torch.Tensor) -> None:
    r"""Draws a pixel into an image.

    Args:
        image: the input image to where to draw the lines with shape :math`(C,H,W)`.
        x: the x coordinate of the pixel.
        y: the y coordinate of the pixel.
        color: the color of the pixel with :math`(C)` where :math`C` is the number of channels of the image.

    Return:
        Nothing is returned.
    """
    image[:, y, x] = color

def draw_line(image: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    r"""Draw a single line into an image.

    Args:
        image: the input image to where to draw the lines with shape :math`(C,H,W)`.
        p1: the start point [x y] of the line with shape (2).
        p2: the end point [x y] of the line with shape (2).
        color: the color of the line with shape :math`(C)` where :math`C` is the number of channels of the image.

    Return:
        the image with containing the line.

    Examples:
        >>> image = torch.zeros(1, 8, 8)
        >>> draw_line(image, torch.tensor([6, 4]), torch.tensor([1, 4]), torch.tensor([255]))
        tensor([[[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0., 255., 255., 255., 255., 255., 255.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]])
    """

    if (len(p1) != 2) or (len(p2) != 2):
        raise ValueError("p1 and p2 must have length 2.")

    if len(image.size()) != 3:
        raise ValueError("image must have 3 dimensions (C,H,W).")

    if color.size(0) != image.size(0):
        raise ValueError("color must have the same number of channels as the image.")

    #IF MY LINE IS OUT OF BOUND IDC JUST DRAW IT TO THE EDGE OF THE IMG
    if (p1[0] >= image.size(2)) or (p1[1] >= image.size(1) or (p1[0] < 0) or (p1[1] < 0)):
        if (p1[0] >= image.size(2)):
            p1[0] = image.size(2) - 1
        if (p1[1] >= image.size(1)):
            p1[1] = image.size(1) - 1
        #raise ValueError("p1 is out of bounds.")

    #IF MY LINE IS OUT OF BOUND IDC JUST DRAW IT TO THE EDGE OF THE IMG
    if (p2[0] >= image.size(2)) or (p2[1] >= image.size(1) or (p2[0] < 0) or (p2[1] < 0)):
        if (p2[0] >= image.size(2)):
            p2[0] = image.size(2) - 1
        if (p2[1] >= image.size(1)):
            p2[1] = image.size(1) - 1
        #raise ValueError("p2 is out of bounds.")

    # move p1 and p2 to the same device as the input image
    # move color to the same device and dtype as the input image
    p1 = p1.to(image.device).to(torch.int64)
    p2 = p2.to(image.device).to(torch.int64)
    color = color.to(image)

    # assign points
    x1, y1 = p1
    x2, y2 = p2

    # calcullate coefficients A,B,C of line
    # from equation Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    if A == 0:
        y2 = y2 + 1
        y1 = y1 - 1
        A = y2-y1
        C = x2 * y1 - x1 * y2
    # make sure A is positive to utilize the function properly
    if A < 0:
        A = -A
        B = -B
        C = -C

    # calculate the slope of the line
    # check for division by zero
    if B != 0:
        m = -A / B

    # make sure you start drawing in the right direction
    x1, x2 = min(x1, x2).long(), max(x1, x2).long()
    y1, y2 = min(y1, y2).long(), max(y1, y2).long()

    # line equation that determines the distance away from the line
    def line_equation(x, y):
        return A * x + B * y + C

    # vertical line
    if B == 0:
        image[:, y1 : y2 + 1, x1] = color
    # horizontal line
    elif A == 0:
        pass
        #A = 0.01
        #y2 = y2+1
        #image[:, y1, x1 : x2 + 1] = color
    # slope between 0 and 1
    elif 0 < m < 1:
        for i in range(x1, x2 + 1):
            _draw_pixel(image, i, y1, color)
            if line_equation(i + 1, y1 + 0.5) > 0:
                y1 += 1
    # slope greater than or equal to 1
    elif m >= 1:
        for j in range(y1, y2 + 1):
            _draw_pixel(image, x1, j, color)
            if line_equation(x1 + 0.5, j + 1) < 0:
                x1 += 1
    # slope less then -1
    elif m <= -1:
        for j in range(y1, y2 + 1):
            _draw_pixel(image, x2, j, color)
            if line_equation(x2 - 0.5, j + 1) > 0:
                x2 -= 1
    # slope between -1 and 0
    elif -1 < m < 0:
        for i in range(x1, x2 + 1):
            _draw_pixel(image, i, y2, color)
            if line_equation(i + 1, y2 - 0.5) > 0:
                y2 -= 1

    return image


def visualise_detections_labels(detections, labels, im):
    """
    im - image
    detections - predicted bounding boxes you wanna draw (red) (x1,y1,x2,y2) - exact pixels on image 
    labels - GT bboxes u wanna draw (green) (x1,y1,x2,y2) - exact pixels on image
    
    only works for one class
    """
    im = im.to(torch.uint8)
    lbls = labels.to(torch.int32)
    dets = detections.to(torch.int32)
    im_drawn = torchvision.utils.draw_bounding_boxes(im, boxes=lbls, labels=[f"{number}" for number in range(len(lbls))], width=6, colors='green', fill=True, font_size=55)
    im_drawn = torchvision.utils.draw_bounding_boxes(im_drawn, boxes=dets, labels=[f"{number}" for number in range(len(dets))], width=2, colors='red', fill=False, font_size=55)
    lines = bboxes_to_lines(dets)
    for line in lines:
        #im_drawn[:, y1, x1 : x2+1] = 
        im_drawn = draw_line(im_drawn,
        torch.tensor([line[0],line[1]]),
        torch.tensor([line[2], line[3]]),
        color=torch.tensor([255,255,2], dtype=torch.uint8))
    a=random.randint(1, 200)
    torchvision.io.write_png(im_drawn, f"slike/test{a}.png")
    torchvision.io.write_png(im.cpu(), f"slike/test{a}_clean .png")
    #kornia.save_image(im_drawn, "test.png")
    return im_drawn

    
def LoGT_loss(detections, labels, im):
    """
    returns a LoGT matrix whether a label has a line going across it

    Args:
        detections (torch.tensor): [Mx4] output of yolo predictions 
        labels (torch.tensor): [Nx4] ground truths in xyxy format
        im (torch.tensor): img CxHxW
    Returns:
        LoGT (torch.tensor): LoGT matrix (LoGT = Line over Ground Truth [N]
    """
    iou = box_iou(labels[:, 1:], detections[:, :4])
    #jiou = jere_iou(best_box_per_label[:, :4], labels[:, 1:], im)
    detections = detections[:,:4]
    labels = labels[:,1:]
    #TODO: best box per label se ne ponasa dobro kada neki GT box nije predictan uopce ?!
    best_box_per_label = detections[torch.argmax(iou, dim=1)]
    #TODO: MAKNI DUPLIKATE IZ BEST BOX PER LABEL
    im_drawn = visualise_detections_labels(best_box_per_label, labels, im)
    lines = bboxes_to_lines(best_box_per_label)
    #assert(detections.shape[0] == lines.shape[0])
    if labels.shape[0] != best_box_per_label.shape[0]:
        LOGGER.info("alo alo falija si jednu ili vise kutija") 
    LoGT = torch.zeros(labels.shape[0], dtype=torch.float32, device=labels.device)
    for i in range(lines.shape[0]):
        line_y = (lines[i][1] + lines[i][3]) / 2
        if labels[i][1] < line_y < labels[i][3]:
        #if line_y > best_box_per_label[i][1] and line_y < best_box_per_label[i][3]:
            LoGT[i] = torch.tensor(1)
            #print("LINE Y IS IN DETECTION")
        else:
            LoGT[i] = torch.tensor(0)
            pass
            #TODO:implementiraj gubitak iou ako je line_y izvan GT bboxa
    a=1
    return LoGT
        
def calculate_logt_on_dataset(logt_on_dataset):
    #TODO: make it better not just 0s and 1s
    pass
    correct_predictions = torch.count_nonzero(logt_on_dataset)
    return (correct_predictions / logt_on_dataset.shape[0]).cpu().item()



def process_batch(detections, labels, iouv, im):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """


    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    #UZET BOX SA NAJVECIN IOU IZ ZA SVAKI LABEL I ONDA NA NJIH PRIMJENIT MOJ ALGO
    #TODO: MOJA METRIKA POKLAPANJA
    #best_box_per_label = detections[torch.argmax(iou, dim=1)]
    #jere_iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                #matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='jupytertest',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 7) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95', 'LoGT')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, logt_list = [], [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    #anchors = model.model.model[-1].anchors
    #print(model.model.model[-1].anchors)
    #LOGGER.info(f'anchori su mi {model.model.model[-1].anchors}')
    #exit()
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
           
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            #scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                #scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv, im[si]*255)
                logt = LoGT_loss(predn.clone(), labelsn.clone(), im[si]*255)
                #logt.to(device)
                altim = im
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            logt_list.append(logt)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls, LoGT tensor)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 100:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)
    #------------END ENTIRE DATALOADER
    # Compute metrics
    pass
    a=1
    for x in zip(*stats):
        a=11
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    a=2
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        logt_on_dataset =  calculate_logt_on_dataset(torch.cat(logt_list))
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 5  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, logt_on_dataset))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('../datasets/coco/annotations/instances_val2017.json'))  # annotations
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, logt_on_dataset, *(loss.cpu() / len(dataloader)).tolist()), maps, t



def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=True)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot

if __name__ =="__main__":
    opt = parse_opt(["--data", "data/police1.yaml", "--weights", "runs/train/yolo5L500ep_25_anchorahardcoded_sgd_finetunepokusaj22/weights/best.pt", "--imgsz", "1024", "--task", "val"])
    main(opt)