import cv2
import numpy as np
from detect import text_recognition
from extract_text import extract_text_box 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pdf_path", type=str,help="path to pdf file")
ap.add_argument("-e", "--extract_type", type=str, default="LTTextBox",
                help="line text recognition type:Line/Box")
ap.add_argument("-t", "--trained_model_path", type=str,help="path to text detection trained model")
ap.add_argument("-c", "--config_path", type=str, help="path to COCO cofig path")
ap.add_argument("-th", "--threshold", type=float, default=0.55,
                help="threshold to determine label/bbox of an instance")
args = vars(ap.parse_args())

pdf_path           = args["pdf_path"]
extract_type       = args["extract_type"]
trained_model_path = args["trained_model_path"]
config_path        = args["config_path"]
threshold          = args["threshold"]
class_names = ['text','title','list','table','figure']

def iou(box1, box2):
    # xmin,ymin,xmax,ymax
    # box1: bboxes of text detection
    # box2: boxxes of line/box text recognition
    
    if (box1[0]>=box2[2]) or (box1[2]<=box2[0]) or \
        (box1[3]<=box2[1]) or (box1[1]>=box2[3]):
         return 0.0
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box1[1])
    xi2 = min(box1[2], box1[2])
    yi2 = min(box1[3], box2[3])
    
    if (yi2-yi1 < 0) or (xi2-xi1 < 0):
        return 0.0
    inter_area = (yi2-yi1)*(xi2-xi1)
    print(box2)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    union_area = box1_area+box2_area-inter_area
    print('inter: ', inter_area)
    print('box1: ', box1_area)
    print('box2: ',box2_area)
    print('union: ',union_area)
    iou_score = inter_area/union_area
    
    return iou_score

# model based
recog_texts, pil_images = text_recognition(pdf_path, trained_model_path, config_path, threshold)
# pdf based
extract_texts           = extract_text_box(pdf_path, extract_type, False)

page1 = cv2.cvtColor(np.array(pil_images[1]), cv2.COLOR_BGR2RGB)
resize_rate = 2
h,w,_ = page1.shape
page1 = cv2.resize(page1, (w//resize_rate, h//resize_rate))
visited = [False]*len(extract_texts[1])

for i, bbox1 in enumerate(recog_texts[1]):
    box1 = bbox1[:4]
    label = class_names[bbox1[5].astype('int')]
    print('class: ', label)
    cv2.rectangle(page1, (int(box1[0]//resize_rate), int(box1[1]//resize_rate)), (int(box1[2]//resize_rate), int(box1[3]//resize_rate)), (0,255,0))
    # cv2.imshow('a',page1)
    # cv2.waitKey(0)
    for j, bbox2 in enumerate(extract_texts[1]):
        box2 = np.array(bbox2[:4])
        print(1)
        iou_score = iou(box1, box2)
        print(iou_score)
        cv2.rectangle(page1, (int(box2[0]//resize_rate), int(box2[1]//resize_rate)), (int(box2[2]//resize_rate), int(box2[3]//resize_rate)), (255,0,0))
        cv2.imshow('a', page1)
        cv2.waitKey(0)
        # print(box1, box2, iou_score, bbox2[-1])
        if (0.0 < iou_score < 1.0) and visited[j] == False:
            # file.write(f'class: {label} \n {bbox2[-1]} \n ------------------------------')
            print(bbox2[-1])
            visited[j] = True
        print("------------------------------")
    page1 = cv2.cvtColor(np.array(pil_images[1]), cv2.COLOR_BGR2RGB)
    page1 = cv2.resize(page1, (w//resize_rate, h//resize_rate))
cv2.destroyAllWindows()

