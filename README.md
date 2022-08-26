## Installation

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' # detectron2 
pip install -r requirements
```

## Note

- Create folder __trained_models__, and download pre-trained models from [https://drive.google.com/drive/folders/10uDDVwzuajsR_BIEVrIZLhUDodyBi5gp?usp=sharing](google drive) and push it into __trained_models__ folder

- File __detect.py__ is to run Detectron2 for Publaynet

- File __extract_text.py__ is to extract bounding box and its text by using PDFMiner 

- File __matching.py__ is to match bounding box to its OCR

## Run

```bash
python detect.py
python extract_text.py
python matching.py -p {pdf_path} -t {trained_model_path} -c {COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml}
```
