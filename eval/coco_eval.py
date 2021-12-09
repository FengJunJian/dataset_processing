from PIL import Image,ImageDraw,ImageFont
import os
import torch
import json
import pickle as plk
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def coco_eval(predictions,dataset,output_folder,iou_type="bbox"):
    '''
    main function for coco evaluation
    predictions: list
    dataset: cocodataset coco(annotation.json)
    output_folder: folder to output the results
    iou_type: enum{'segm' ,'bbox', 'keypoints'}
    '''

    coco_boxes=prepare_for_coco_detection(predictions=predictions,dataset=dataset)
    file_path= os.path.join(output_folder, iou_type + ".json")
    with open(file_path, "w") as f:
        json.dump(coco_boxes, f)
    coco_dt = dataset.coco.loadRes(str(file_path)) if file_path else COCO()
    coco_gt=dataset.coco
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # p_a1 = coco_eval.eval['precision'][0, :, 0, 0, 2]  # (T, R, K, A, M)
    #r_a1 = coco_eval.eval['recall'][0, 0, 0, 2]  # (T, K, A, M)
    # pr_array2 = res.eval['precision'][2, :, 0, 0, 2]
    # pr_array3 = res.eval['precision'][4, :, 0, 0, 2]
    # r_a1 = np.arange(0.0, 1.01, 0.01)
    #pr_c=[]
    pr_c={'total':coco_eval.eval}
    for catId in coco_gt.getCatIds():#各类AP
        coco_eval_c = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval_c.params.catIds = [catId]
        coco_eval_c.evaluate()
        coco_eval_c.accumulate()
        #coco_eval_c.summarize()
        #pr_c.append(coco_eval_c.eval)
        pr_c[catId]=coco_eval_c.eval

    if output_folder:
        with open(os.path.join(output_folder,"coco_PR_all.pkl"),'wb') as f:
            plk.dump(pr_c,f)
        # with open(os.path.join(output_folder,"coco_results.txt"),'w') as f:
        #     for k,v in results.results.items():
        #         if isinstance(v,dict):
        #             for k1,v1 in v.items():
        #                 f.write(str(k1)+'\t'+str(v1)+'\n')
        # for iou_type in iou_types:
        #     with open(os.path.join(output_folder,iou_type+"PR.txt"),'w') as f:
        #         for d1,d2 in zip(x,p_a1):
        #             f.write(str(d1)+'\t'+str(d2)+'\n')
    # pp=coco_eval_c.eval['precision'][0, :, 0, 0, 2]
    # rr = np.arange(0.0, 1.01, 0.01)
    # voc_ap(rr,pp,False)
    # T = len(p.iouThrs)
    # R = len(p.recThrs)
    # K = len(p.catIds) if p.useCats else 1
    # A = len(p.areaRng)
    # M = len(p.maxDets)
    # precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    # recall = -np.ones((T, K, A, M))
    # scores = -np.ones((T, R, K, A, M))
    #T:10 iouThrs    - [.5:.05:.95]
    #R:101 recThrs    - [0:.01:1]
    #K:number of categories
    #A:4, object area ranges,[[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]->[all,small,medium,large]
    #M:3 thresholds on max detections per image, [1 10 100]
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    return coco_eval

if __name__=='__main__':
    json_file='E:/fjj/SeaShips_SMD/voc_cocostyletest.json'#gt
    res_file='E:/SSL/ssl_detection-master/detection/eval.json-voc_cocostyletest'#predict
    coco = COCO(json_file)#return gt class COCO
    '''
    res_file:eg. list for predicted_boxes
    'image_id': 1,
     'category_id': 2,
     'bbox': [610.029, 355.865, 748.923, 71.877],
     'score': 0.9703}
    '''
    cocoDt=coco.loadRes(res_file)#return predicted box class COCO

    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()