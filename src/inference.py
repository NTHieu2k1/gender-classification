import base64
import os
import torch
import json
import numpy as np
import cv2 as cv
import logging
from torch import nn
from torchvision import models
from torchvision import transforms as T

MODEL_NAME = 'gender_classifier_250418.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    # Base model
    logger.info('model_fn: Loading the model')
    model = models.vgg13(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(7*7*512, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1)
    )
    with open(os.path.join(model_dir, MODEL_NAME), 'rb') as model_loader:
        model.load_state_dict(torch.load(model_loader, map_location=device))
    model.to(device).eval()
    logger.info('model_fn: Loading model done')
    return model


def input_fn(request_body, content_type='application/json'):
    if content_type != 'application/json':
        logger.error(f'input_fn: Unsupported request content type: {content_type}')
        raise ValueError(f'Unsupported request content type: {content_type}')
    # Decode image data
    logger.info('input_fn: Decode image data')
    json_data = json.loads(request_body)
    image_encoded = json_data['image']
    image_raw = base64.b64decode(image_encoded)
    # Load image from decoded bytestring
    logger.info('input_fn: Load image from decoded byte string')
    image_bytes = np.frombuffer(image_raw, np.uint8)
    image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Transform image
    logger.info('input_fn: Apply transforms')
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)[None].to(device)
    logger.info('input_fn: Transform complete')
    return image


def predict_fn(input_data, model):
    logger.info('predict_fn: Feeding to the model')
    with torch.inference_mode():
        logits = model(input_data)
        preds = torch.sigmoid(logits).cpu().item()
    logger.info('predict_fn: Prediction generated')
    return preds


def output_fn(prediction, content_type='application/json'):
    if content_type != 'application/json':
        logger.error(f'output_fn: Unsupported request content type: {content_type}')
        raise ValueError(f'Unsupported request content type: {content_type}')
    logger.info('output_fn: Processing predictions')
    if prediction >= 0.5:
        content = dict(pred='Male', confidence=prediction)
    else:
        content = dict(pred='Female', confidence=1-prediction)
    logger.info('output_fn: Prediction formatted')
    return json.dumps(content)
