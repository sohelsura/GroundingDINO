# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
import shutil
from datetime import datetime
import io
import os
import cv2
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import torch
from PIL import Image
import numpy as np
from PIL import Image as Img
from typing import List
import warnings
import uuid
import aiohttp
from io import BytesIO

app = FastAPI()

current_directory = os.path.dirname(__file__)


UPLOAD_DIR = f"{current_directory}/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


warnings.filterwarnings("ignore")

current_directory = os.path.dirname(__file__)

GROUNDING_DINO_CONFIG_PATH = f"{current_directory}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = f"{current_directory}/weights/groundingdino_swint_ogc.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grounding_dino_model = None

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = f"{current_directory}/weights/sam_vit_h_4b8939.pth"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def load_grounding_dino_model():
    global grounding_dino_model
    if grounding_dino_model is None:
        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        print("Grounding DINO model loaded successfully")
    else:
        print("Grounding DINO model already loaded")

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]

def auto_crop(image_path):
    global grounding_dino_model
    # Load the model if not loaded
    load_grounding_dino_model()
    
    image2 = Image.open(image_path)


    SOURCE_IMAGE_PATH = image_path
    CLASSES = ['hair']
    BOX_TRESHOLD = 0.40
    TEXT_TRESHOLD = 0.25
    image = cv2.imread(SOURCE_IMAGE_PATH)
    # image = output_chbg_scaled

    # Detect objects and annotate the image
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    for x1, y1, x2, y2 in detections.xyxy:
        # Extract the region of interest using bounding box coordinates
        print(y2)
        if y2 < 195:
            y2 = y2 + 50
            x1 = x1 - 20
            x2 = x2 + 20
        x1, y1, x2, y2_h = int(x1), int(y1), int(x2), int(y2)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
        cutout = image[y1:y2_h, x1:x2]
    
    # print(f"This is the cutout {cutout}")

    # imgname , extension = os.path.splitext(os.path.basename(image_path))
    # output_path = os.path.join('/home/ubuntu/tmp/GroundingDINO/uploads' , f'{str(uuid.uuid4())}.png')
    # cv2.imwrite(output_path, cutout)
    # print(f"cutout {output_path}")

    CLASSES = ['head']
    BOX_TRESHOLD = 0.42
    TEXT_TRESHOLD = 0.35

    image = cv2.imread(SOURCE_IMAGE_PATH)
    # image = output_chbg_scaled

    # Detect objects and annotate the image
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    for box in detections.xyxy:
        global new_cutout
        x1, y1, x2, y2 = box
        # Extract the region of interest using bounding box coordinates
        print(y2)
        y2 = y2 + 10
        x1 = x1 - 20
        x2 = x2 + 20
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
        # cutout = image[y1:y2, x1:x2]
        new_cutout = image[y1:y2, x1:x2]

    if y2_h < y2:
        cutout = new_cutout       
        # print(f"This is the cutout {cutout}")
        # imgname , extension = os.path.splitext(os.path.basename(image_path))
        # output_path = os.path.join('/home/ubuntu/tmp/GroundingDINO/uploads' , f'{str(uuid.uuid4())}.png')
        # cv2.imwrite(output_path, cutout)
        # print(f"cutout {output_path}")
    
    _, buffer = cv2.imencode('.png', cutout)
    return io.BytesIO(buffer)

    return output_path

def hair_mask(source_image_path: str) -> str:

    load_grounding_dino_model()

    CLASSES = ['hair']
    BOX_THRESHOLD = 0.40
    TEXT_THRESHOLD = 0.25

    image = cv2.imread(source_image_path)
    # Detect objects and annotate the image
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    
    base_name = os.path.splitext(os.path.basename(source_image_path))[0]
    mask_pil = Image.fromarray(detections.mask[0])
    
    mask_bytes_io = io.BytesIO()
    mask_pil.save(mask_bytes_io, format='JPEG')
    mask_bytes_io.seek(0)  # Reset buffer position to the beginning

    return mask_bytes_io

    mask_path = f"/home/ubuntu/tmp/arjundheek/static/uploads/mask/{str(uuid.uuid4())}_hair_mask.jpg"
    mask_pil.save(mask_path)
    

    return mask_path

async def fetch_image_from_url(url: str) -> io.BytesIO:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid URL")
            content = await response.read()
            return BytesIO(content)
        
@app.post("/generate-crop-image/")
async def auto_crop_endpoint(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image file or URL provided")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    else:
        image_io = await fetch_image_from_url(url)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

    cropped_image_io = auto_crop(image_path)
    cropped_image_io.seek(0)

    os.remove(image_path)

    return StreamingResponse(cropped_image_io, media_type="image/png")

@app.post("/generate-hair-mask/")
async def auto_hair_mask_endpoint(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image file or URL provided")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    else:
        image_io = await fetch_image_from_url(url)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

    mask_io = hair_mask(image_path)
    mask_io.seek(0)

    os.remove(image_path)

    return StreamingResponse(mask_io, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
