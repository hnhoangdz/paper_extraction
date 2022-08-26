# -*- coding: utf8 -*-
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdf2image import convert_from_path
import pdfminer
import cv2
import numpy as np

def pdfminer_extract(path,param):
    if "LTTextBox" in param:
        param = pdfminer.layout.LTTextBox
    elif "LTTextLine" in param:
        param = pdfminer.layout.LTTextLine
    else:
        assert False,"False"
    fp = open(path, 'rb')

    # Create a PDF parser object associated with the file object.
    parser = PDFParser(fp)

    # Create a PDF document object that stores the document structure.
    # Password for initialization as 2nd parameter
    document = PDFDocument(parser)

    # Check if the document allows text extraction. If not, abort.
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()

    # Create a PDF device object.
    device = PDFDevice(rsrcmgr)

    # BEGIN LAYOUT ANALYSIS
    # Set parameters for analysis.
    laparams = LAParams()

    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)

    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    def parse_obj(lt_objs):
        arr = []
        # loop over the object list
        for obj in lt_objs: 
            # if it's a textbox, print text and location
            if isinstance(obj, param):
                text = obj.get_text().replace('\n', ' ')
                text_clean = text.strip()

                start_index = text.find(text_clean)
                end_index = start_index + len(text_clean)
                if start_index > 0:
                    start_index-=1
                left, upper, right,lower = obj.bbox

                step = (right-left) / len(text)
                right = left + step*end_index
                left = left + step*start_index
                arr.append([left, upper, right,lower, obj.get_text()])

            # if it's a container, recurse
            else:
                try:
                    tmp_arr = parse_obj(obj._objs)
                    arr.extend(tmp_arr)
                except:
                    pass
        return arr
    result = []
    # loop over all pages in the document
    sizes = None
    for page in PDFPage.create_pages(document):
        if sizes is None:
            sizes = page.mediabox
           
        # read the page into a layout object
        interpreter.process_page(page)
        layout = device.get_result()
        # extract text from this object
        result.append(parse_obj(layout._objs))
        
    return result,sizes

def extract_text_box(pdf_path, extract_type="LTTextBox", visualize=True):
    pred_boxes = []
    try:
        boxes,sizes = pdfminer_extract(pdf_path, extract_type)
        images = convert_from_path(pdf_path)

        if len(images)!= len(boxes):
            return None
        for i in range(len(images)):
            
            boxs_perI = []
            image = np.array(images[i])
            h,w,_ = image.shape
            x_scale = w/sizes[2]    
            y_scale = h/sizes[3]

            results = []
            for box in boxes[i]:
                if len(box[4].replace("\xa0","").strip())<1:
                    continue
                # xmin, ymin, xmax, ymax
                results.append([box[0],sizes[3] - box[1],box[2],sizes[3] - box[3],box[4].replace("\xa0","")])

            if len(results) == 0:
                continue
            
            for box in results:
                
                xmin, ymax, xmax, ymin, text  = box
                
                xmin -= 5
                xmax += 5
                    
                xmin,xmax = x_scale*xmin, x_scale*xmax
                ymin,ymax = y_scale*ymin, y_scale*ymax
                if visualize:
                    print(h, w)
                    print(xmin, ymin, xmax, ymax, text)
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0))
                    cv2.imshow('a', image)
                    cv2.waitKey()
                boxs_perI.append((xmin, ymin, xmax, ymax, text))
                
            pred_boxes.append(boxs_perI)
        if visualize:
            cv2.destroyAllWindows(0)
    except Exception as e:
        print(e,pdf_path)
        return None
    return pred_boxes

if __name__ == "__main__" :
    bbox_text = extract_text_box("20220190.pdf")
    print(bbox_text)