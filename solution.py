from typing import List, Union
from ultralytics import YOLO
import logging
import os
#import time
import numpy as np
from typing import List, Optional


from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.slicing import slice_image

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}


num_batch = 1
merge_buffer_length = None
perform_standard_pred = True
postprocess_class_agnostic: bool = False
model = YOLO('best.pt')
model.export(format='engine', device='cuda:0', half=True, imgsz=640, nms=True, verbose=False)
model_trt  = YOLO('best.engine', task='detect')




def predict(images: Union[List[np.ndarray], np.ndarray],
            model_confidence=0.15,
            overlaping=0.3,
            half_=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric='IOS',
            postprocess_match_threshold=0.15,
            side_size=640,
            iou_=0.7) -> dict:

    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic
    )

    if isinstance(images, np.ndarray):
        images = [images]

    all_predictions = []

    for image in images:
        image = image[:, :, ::-1]
        slice_image_result = slice_image(
            image=image,
            slice_height=side_size,
            slice_width=side_size,
            overlap_height_ratio=overlaping,
            overlap_width_ratio=overlaping,
            auto_slice_resolution=False
        )

        num_slices = len(slice_image_result)
        num_group = int(num_slices / num_batch)
        object_prediction_list = []

        for group_ind in range(num_group):
            image_list = []
            shift_amount_list = []
            for image_ind in range(num_batch):
                slice_img = slice_image_result.images[group_ind * num_batch + image_ind]
                shift = slice_image_result.starting_pixels[group_ind * num_batch + image_ind]
                image_list.append(slice_img)
                shift_amount_list.append(shift)

            res = model_trt.predict(
                image_list[0],
                verbose=False,
                conf=model_confidence,
                device='cuda:0',
                half=half_,
                imgsz=side_size,
                iou=iou_,
                augment=False
            )

            for r, shift in zip(res, shift_amount_list):
                shift_x, shift_y = shift
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    # Применяем сдвиг
                    x1 += shift_x
                    x2 += shift_x
                    y1 += shift_y
                    y2 += shift_y
                    object_prediction_list.append(
                        ObjectPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=conf,
                            category_id=0
                        )
                    )

            if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
                object_prediction_list = postprocess(object_prediction_list)

        if num_slices > 1 and perform_standard_pred:
            res = model_trt.predict(
                image,
                verbose=False,
                conf=model_confidence,
                device='cuda:0',
                half=half_,
                iou=iou_,
                augment=False
            )
            for r in res:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    object_prediction_list.append(
                        ObjectPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=conf,
                            category_id=0
                        )
                    )

        if len(object_prediction_list) > 1:
            object_prediction_list = postprocess(object_prediction_list)

        pp = PredictionResult(
            image=image,
            object_prediction_list=object_prediction_list
        )

        predictions_list = []
        for obj in pp.object_prediction_list:
            x_min, y_min, x_max, y_max = obj.bbox.to_voc_bbox()
            xc = (x_min + x_max) / 2.0 / image.shape[1]
            yc = (y_min + y_max) / 2.0 / image.shape[0]
            w = (x_max - x_min) / image.shape[1]
            h = (y_max - y_min) / image.shape[0]
            conf = obj.score.value

            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                continue

            predictions_list.append({
                'xc': round(xc, 4),
                'yc': round(yc, 4),
                'w': round(w, 4),
                'h': round(h, 4),
                'label': 0,
                'score': round(conf, 4)
            })

        all_predictions.append(predictions_list)

    return all_predictions
