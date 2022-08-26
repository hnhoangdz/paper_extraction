import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pdf2image import convert_from_path

def text_recognition(pdf_path, 
                    trained_model_path, 
                    config_path,
                    threshold=0.55,
                    class_names=['text','title','list','table','figure'],
                    colors = {
                        'text': (0, 0, 255), 
                        'title': (255, 0, 0),
                        'list': (0, 255, 0),
                        'table': (255, 128, 0),
                        'figure': (255, 153, 255)
                    }):
    
    # Convert pdf to images
    images = convert_from_path(pdf_path)

    # Detectron config
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.WEIGHTS = trained_model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

    # Detectron predictor
    predictor = DefaultPredictor(cfg)

    results = []
    for i, image_pil in enumerate(images):
        
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        outputs = predictor(image_cv)

        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        for j in range(0, len(pred_boxes)):
            box = pred_boxes[j]
            # print(box)
            score = round(float(scores[j]), 4)
            label_key = int(pred_classes[j])
            label = class_names[label_key]
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            cv2.rectangle(image_cv, (xmin,ymin), (xmax, ymax), colors[label])
            cv2.putText(image_cv, label + ":" + str(score), \
                        (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX,\
                        1, colors[label], 2, cv2.LINE_AA)
        cv2.imwrite(f'page{i}.jpg', image_cv)
        
        scores = np.expand_dims(scores, axis=1)
        pred_classes = np.expand_dims(pred_classes, axis=1)
        
        # [x,y,w,h,score,label]
        res = np.concatenate((pred_boxes, scores, pred_classes), axis=1)
        
        # a page
        results.append(res)

    return results, images
    
if __name__ == "__main__":
    pdf_path = '20220190.pdf'
    trained_model_path = 'trained_models/model_final.pth'
    config_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    results = text_recognition(pdf_path, trained_model_path, config_path)
    print(results[0])