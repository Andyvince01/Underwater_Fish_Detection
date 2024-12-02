# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

# from tensordict import TensorDict

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.results import Boxes
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops, RESIZERS, SECOND_ANALYZERS, VOC
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


# RESIZERS AND SECOND ANALYZER FUNCTIONS ---------------------------------------------------------------#
from ultralytics.utils.resizers import ResizerConfig, ResizerFactory
RESIZERS_CFG = [
    ResizerConfig(
        type=resizer_type,
        **resizer_cfg
    )
    for resizer_type, resizer_cfg in RESIZERS.items()
] if RESIZERS else []

from ultralytics.engine.vlm import voc_in_caption, answers_yes

SECOND_ANALYZERS_MAP = {
    'answers_yes': answers_yes,
    'voc_in_caption': voc_in_caption
}

class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        self.avg_iou = []
        self.mae = []
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )
    
        # resizer --------------------------------------------------------------------------------------------#
        self.resizer = None
        if RESIZERS_CFG:
            self.resizer = list({config.type.name.lower(): ResizerFactory.create(config) for config in RESIZERS_CFG}.values())[0]
            LOGGER.info(f"Resizer: {self.resizer}") if self.resizer else None 
        # second analyzers -----------------------------------------------------------------------------------#
        self.second_analyzer = None
        if SECOND_ANALYZERS:
            second_analyzer_name = list(SECOND_ANALYZERS.keys())[0]
            self.params = list(SECOND_ANALYZERS.values())[0] if second_analyzer_name == 'answers_yes' else {'vocabulary': VOC}
            self.second_analyzer = lambda frame : SECOND_ANALYZERS_MAP.get(second_analyzer_name, None)(frame, **self.params)
            LOGGER.info(f"Second Analyzer: {second_analyzer_name} | Params: {self.params}") if self.second_analyzer else None
        
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        
        self.size_list = ['overall', 'small', 'medium', 'large'] if not self.training else ['overall']
        self.stats = {k : dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[]) for k in self.size_list}

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%11s" + "%11s" + "%11s" * 6) % ("Size", "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        im_file = batch["im_file"][si]
        img = batch['img'][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad, "im_file": im_file, 'img': img}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = {k : dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            ) for k in self.size_list}
            
            # Prepare batch ---------------------------------------------------------------------------------#
            pbatch = self._prepare_batch(si, batch)
               
            # Get current image information ---------------------------------------------------------------#         
            img, cls, bbox = pbatch.pop('img').squeeze().cpu().numpy(), pbatch.pop("cls"), pbatch.pop("bbox")
            img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
            
            nl = len(cls)
            for size_name in self.size_list:
                stat[size_name]["target_cls"] = cls
                stat[size_name]["target_img"] = cls.unique()

            if npr == 0:
                if nl:
                    for size_name in self.size_list:
                        for k in stat[size_name].keys():
                            self.stats[size_name][k].append(stat[size_name][k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)

            for size_name in self.size_list:
                stat[size_name]['conf'] = predn[:, 4]
                stat[size_name]["pred_cls"] = predn[:, 5]                                             
            
            # Apply Second Analyzer ---------------------------------------------------------------------#
            if nl > 0 and (self.second_analyzer is not None or self.resizer is not None):
                # Convert the bbox tensors to Boxes
                predicted_boxes = Boxes(predn, pbatch["ori_shape"])
                
                # Iterate over the predicted boxes ------------------------------------------------------#
                bboxes = []
                for box in predicted_boxes:
                    # If conf is above 0.8, not apply the second analyzer ---------------------------------#
                    if box.conf.data > 0.75: bboxes.append(box.data); continue
                    
                    # Apply the resizer to the original frame and the bounding box -------------------------------------------------#
                    resized_frame = self.resizer(img, box)
                                                
                    # Apply the second analyzer to the resized frame and the bounding box -------------------------------------------------#
                    status = self.second_analyzer(resized_frame)
                                                            
                    # If the status is True, add the bounding box to the list of bounding boxes -------------------------------------------------#
                    if status: bboxes.append(box.data)
                    else: continue
                    
                # Convert the list of bounding boxes to a tensor -------------------------------------------------#
                predn = torch.cat(bboxes, dim=0) if len(bboxes) > 0 else torch.zeros((0, 6), device=self.device)           

            # Evaluate
            if nl:
                results, detections_size, gt_cls_size = self._process_batch(predn, bbox, cls)
                for size_name in self.size_list:
                    stat[size_name]['tp'] = results[size_name]; stat[size_name]['conf'] = detections_size[size_name][:, 4]; 
                    stat[size_name]['pred_cls'] = detections_size[size_name][:, 5]; stat[size_name]["target_cls"] = gt_cls_size[size_name]; 
                    stat[size_name]["target_img"] = gt_cls_size[size_name].unique()                                                           
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for size_name in self.size_list:
                for k in stat[size_name].keys():
                    self.stats[size_name][k].append(stat[size_name][k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )
                
    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {}
        
        for size_name, size_value in self.stats.items():
            stats[size_name] = {k: torch.cat(v, 0).cpu().numpy() for k, v in size_value.items()}
                        
        self.nt_per_class = {size_name: np.bincount(stats[size_name]["target_cls"].astype(int), minlength=self.nc) for size_name in self.size_list}
        self.nt_per_image = {size_name: np.bincount(stats[size_name]["target_img"].astype(int), minlength=self.nc) for size_name in self.size_list}
        for size_name in self.size_list:
            stats[size_name].pop("target_img", None)
                                            
        if len(stats):
            for size_name in self.size_list:
                if stats[size_name]["tp"].any():
                    self.metrics.process(
                        tp=stats[size_name].get("tp"),
                        conf=stats[size_name].get("conf"),
                        pred_cls=stats[size_name].get("pred_cls"),
                        target_cls=stats[size_name].get("target_cls"),
                        size_name=size_name
                    )
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%11s" + "%11s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        for size_name in self.size_list:
            self.metrics.size_name = size_name
            LOGGER.info(pf % (size_name, "all", self.seen, self.nt_per_class[size_name].sum(), *self.metrics.mean_results(key=size_name)))
            if self.nt_per_class[size_name].sum() == 0:
                LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

            # Print results per class
            if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
                for i, c in enumerate(self.metrics.ap_class_index):
                    LOGGER.info(
                        pf % (size_name, self.names[c], self.nt_per_image[size_name][c], self.nt_per_class[size_name][c], *self.metrics.class_result(i, key=size_name))
                    )
                    
        LOGGER.info("--- Average IOU: %.3f" % np.mean(self.avg_iou))
        LOGGER.info(f"--- Mean Absolute Error: {np.mean(self.mae):.3f} | Stddev: {np.std(self.mae):.3f}")
        
        # for k in range(10):
        #     LOGGER.info(f"AP @{(k)*5 + 50:.2f}: {self.metrics.box.all_ap[:, k].mean():.3f}")
        #     for i, c in enumerate(self.metrics.ap_class_index):
        #         LOGGER.info(f"--- Class: {self.names[c]}: {self.metrics.box.all_ap[i, k]:.3f}")

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
        
        LOGGER.info("---"*30)


    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        #--- Initialize results dictionary ---#
        detections_size = {k : None for k in self.size_list}
        gt_cls_size = {k : None for k in self.size_list}
        results = {k : None for k in self.size_list}
        
        #--- All results ---#
        iou_all = box_iou(gt_bboxes, detections[:, :4])

        detections_size['overall'] = detections
        gt_cls_size['overall'] = gt_cls
        results['overall'] = self.match_predictions(detections[:, 5], gt_cls, iou_all)
        
        # Calculate the coordinates of the intersection rectangle
        max_ious = torch.max(iou_all, dim=1)[0]
        mean_ious = max_ious.mean().item()
        self.avg_iou.append(mean_ious)
        self.mae.append(abs(gt_bboxes.shape[0] - detections.shape[0]))
                
        results['overall'] = self.match_predictions(detections[:, 5], gt_cls, iou_all)

        if self.training:
            return results, detections_size, gt_cls_size

        #--- Match box labels (small 0, medium 1, large 2) to gt boxes ---#
        gt_widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        gt_area = gt_widths * gt_heights
                    
        #--- Match box labels (small 0, medium 1, large 2) to detections ---#
        det_widths = detections[:, 2] - detections[:, 0]
        det_heights = detections[:, 3] - detections[:, 1]
        det_area = det_widths * det_heights
        
        # Assign size labels to detections
        gt_areaRngLbl = torch.where(gt_area < 32*32, 0, torch.where((gt_area >= 32*32) & (gt_area < 96*96), 1, 2))
        det_areaRngLbl = torch.where(det_area < 32*32, 0, torch.where((det_area >= 32*32) & (det_area < 96*96), 1, 2))

        #--- Extract IoU and classes for each size category ---#
        size_map = {'small': 0, 'medium': 1, 'large': 2}
        for size_name, size_value in size_map.items():
            # Extract gt boxes for the current size category
            gt_bboxes_size = gt_bboxes[gt_areaRngLbl == size_value]

            # Extract gt classes for the current size category
            gt_cls_size[size_name] = gt_cls[gt_areaRngLbl == size_value]

            # Extract detections for the current size category
            detections_size[size_name] = detections[det_areaRngLbl == size_value]
                            
            # Calculate IoU for each size category
            iou_size = box_iou(gt_bboxes_size, detections_size[size_name][:, :4])
                                
            # Match predictions for each size category
            results[size_name] = self.match_predictions(detections_size[size_name][:, 5], gt_cls_size[size_name], iou_size)
            
        #--- Return results ---#
        return results, detections_size, gt_cls_size
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats