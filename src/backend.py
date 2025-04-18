import torch
from torch import nn
from torchvision import models
from torchvision import transforms as T

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
        nn.Linear(7*7*512, 2048),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 1)
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
