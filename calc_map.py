import sys, os
sys.path.append("/home/ubuntu/workspace/ywshin/construct/yolov5")
from utils.torch_utils import select_device
from utils.general import scale_coords, coco80_to_coco91_class, LOGGER
from utils.metrics import box_iou, ap_per_class
import torch, json
import numpy as np
import copy
import hydra
from tqdm import tqdm

def process_batch(detections, labels, iouv, new_coco_idx, device):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)    
    iou = box_iou(labels[:, 1:], detections[:, :4])
    
#     print(iou)
    new_cls = [float(new_coco_idx[int(a)]) for a in detections[:, 5]]
    # np.array(new_ds)
    det_cls = torch.tensor(new_cls).to(device)
    
#     print((labels[:, 0:1] == det_cls))
    
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == det_cls))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = copy.deepcopy(x)
    y[0] = x[0]  # top left x
    y[1] = x[1]  # top left y
    y[2] = x[0] + x[2]  # bottom right x
    y[3] = x[1] + x[3]  # bottom right y
    return y

def get_annos(img_id, coco_gts):
    annos = []
    for anno in coco_gts:
        if anno["image_id"] == img_id:
            annos.append([anno['category_id']] + xywh2xyxy(anno['bbox']))
    return annos

@hydra.main(config_path="conf", config_name="eval")
def calc_map(cfg):
#     Load Ground Truth
    with open(cfg.gt_path, 'r') as f:
        coco = json.load(f)

    coco_cat = coco["categories"]
    coco_names = {}
    for i, c in enumerate(coco_cat):
        coco_names[c["id"]] = c["name"]

    new_coco_idx = coco80_to_coco91_class()
    nc = cfg.nc
    coco_gts = coco["annotations"]

#     Load predicted values
    with open(cfg.target_path, 'r') as f:
        results = json.load(f)

#     set eval parameters
    device = cfg.device
    device = select_device(device, batch_size=32)

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    
    stats = []
    seen = len(results)
    for result in tqdm(results):
        img_path = result["img_name"]
        pred = result["pred"]
        npr = np.shape(pred)[0]
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
        det_cls = torch.tensor([]).to(device)

        bname = os.path.basename(img_path)
        img_id = int(bname[:-4])

        annos = get_annos(img_id, coco_gts)
        annosn = torch.tensor(annos).to(device)

        predn = torch.tensor(pred).to(device)

        if npr == 0:
            if len(annos) > 0:
                stats.append((correct, *torch.zeros((3, 0), device=device)))
            continue
        
        new_cls = [float(new_coco_idx[int(a)]) for a in predn[:, 5]]
        # np.array(new_ds)
        det_cls = torch.tensor(new_cls).to(device)
        
        if len(annos) > 0:
            correct = process_batch(predn, annosn, iouv, new_coco_idx, device)
            stats.append((correct, predn[:, 4], det_cls, annosn[:, 0]))
        else:
            stats.append((correct, predn[:, 4], det_cls, *torch.zeros((1, 0), device=device)))
    
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] 
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="", names=coco_names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        
    cpf = '%20s' + '%11s' * 2 + '%11.3s' * 2 + '%9.3s' + '%0i' + '%11.3s' # print format
    LOGGER.info(cpf % ('Category', "Image #", "count", "mp", "mr", "map", 50, "map"))

    
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    for i, c in enumerate(ap_class):
        LOGGER.info(pf % (coco_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        
if __name__ == '__main__':
    calc_map()
    
    
    