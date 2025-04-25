import torch
from torch import nn
from torchvision import models
from torchvision import transforms as T
from io import BytesIO
import requests
import boto3
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path):
    # Image transform
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Base model
    model = models.vgg13(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(7*7*512, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1)
    )
    if device == 'cpu':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif device == 'cuda':
        model.load_state_dict(torch.load(checkpoint_path))
    return model, transform


def classify(image, model, transform, return_proba=False):
    # Convert the input image to the right form
    image = transform(image)[None].to(device)
    # Feed into the model
    with torch.inference_mode():
        logits = model(image)
        pred = torch.sigmoid(logits).cpu().item()
        if pred >= 0.5:
            pred_class = 'Male'
            pred_prob = pred
        else:
            pred_class = 'Female'
            pred_prob = 1 - pred
    return pred_class, pred_prob if return_proba else None


def upload_n_classify(request_data):
    runtime = boto3.client('sagemaker-runtime')
    endpoint_name = 'gender-classifier'
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=request_data
    )
    result = json.loads(response['Body'].read().decode())
    return result






