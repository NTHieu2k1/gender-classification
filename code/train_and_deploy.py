import os
import numpy as np
import pandas as pd
import torch
import cv2 as cv
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms as T
from argparse import ArgumentParser
import zipfile
from torchmetrics import Accuracy
from torchinfo import summary
from torch_snippets.torch_loader import Report
import logging
import base64
import json

logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--model_filename', type=str, default='model.pt')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Define Dataset class
class GenderDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        super().__init__()
        self.data_dir = data_dir
        csv_path = os.path.join(self.data_dir, csv_file)
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_image(self, idx):
        path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.df.iloc[idx, 1]
        label = torch.tensor([1.0]) if label == 'male' else torch.tensor([0.0])    # 0: Female, 1: Male
        if self.transform:
            image = self.transform(image)
        return image.to(device), label.to(device)


# Define callback for early stopping (to minimize overfitting)
class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.best_score = -1
        self.best_model_state = None
        self.mode = mode
        self.counter = 0

    def __call__(self, model_score, model):
        score = -model_score if self.mode == 'min' else model_score
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def get_best_model(self):
        return self.best_model_state


# Define helper functions
def prepare_for_training(model: nn.Module, lr: float, device: torch.device = device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    accuracy = Accuracy(task='binary').to(device)
    return criterion, optimizer, accuracy

def train_step(model: nn.Module, batch, criterion: nn.Module, optimizer: optim.Optimizer, accuracy, device: torch.device = device):
    model.train()
    X, y = batch
    y_hat = model(X)
    loss = criterion(y_hat, y)
    acc = accuracy(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

def test_step(model: nn.Module, batch, criterion: nn.Module, accuracy, device: torch.device = device):
    model.eval()
    with torch.inference_mode():
        X, y = batch
        y_hat = model(X)
        loss = criterion(y_hat, y)
        acc = accuracy(y_hat, y)
    return loss.item(), acc.item()

def fit(model, train_loader, test_loader, criterion, optimizer, accuracy, epochs, device=device, **kwargs):
    log = Report(epochs)
    early_stopping = EarlyStopping(patience=3, mode='max')
    for epoch in range(epochs):
        N = len(train_loader)
        for idx, batch in enumerate(train_loader):
            train_loss, train_acc = train_step(model, batch, criterion, optimizer, accuracy, device)
            log.record(epoch + (idx+1)/N, train_loss=train_loss, train_acc=train_acc, end='\r')
        N = len(test_loader)
        for idx, batch in enumerate(test_loader):
            test_loss, test_acc = test_step(model, batch, criterion, accuracy, device)
            log.record(epoch + (idx+1)/N, test_loss=test_loss, test_acc=test_acc, end='\r')
        avgs = log.report_avgs(epoch+1)
        early_stopping(avgs['epoch_test_acc'], model)
        if early_stopping.early_stop:
            break
    # Load the best model
    model.load_state_dict(early_stopping.best_model_state)


# Unzip data zip file
with zipfile.ZipFile(f'{args.train}/train.zip') as train_file:
    train_file.extractall(args.train)
with zipfile.ZipFile(f'{args.test}/test.zip') as test_file:
    test_file.extractall(args.test)

# Define VGG11_BN like transform
vgg11_bn_transform = T.Compose([
    T.ToTensor(),
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load to Dataset -> DataLoader instances
train_ds = GenderDataset(args.train, 'train.csv', vgg11_bn_transform)
test_ds = GenderDataset(args.test, 'test.csv', vgg11_bn_transform)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size)

# Build model based on pretrained VGG11_BN architecture
vgg11_bn_weights = models.VGG11_BN_Weights.DEFAULT
model = models.vgg11_bn(weights=vgg11_bn_weights)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(7*7*512, 2048),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(2048, 1)
)
model.to(device)
summary(model, input_size=[32, 3, 224, 224], col_names=['input_size', 'output_size', 'num_params', 'trainable'], col_width=20,
        row_settings=['var_names'])

# Setup for training
criterion, optimizer, accuracy = prepare_for_training(model, lr=args.learning_rate)

# Training
fit(model, train_loader, test_loader, criterion, optimizer, accuracy, epochs=args.epochs)

# Save the trained model
torch.save(model.state_dict(), f=f'{args.model_dir}/{args.model_filename}')


# Define functions for inference
def model_fn(model_dir):
    # Base model
    logger.info('model_fn: Loading the model')
    model = models.vgg11_bn(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(7*7*512, 2048),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 1)
    )
    with open(os.path.join(model_dir, MODEL_NAME), 'rb') as model_loader:
        model.load_state_dict(torch.load(model_loader, map_location=device))
    model.to(device).eval()
    return model

def input_fn(request_body, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError(f'Unsupported request content type: {content_type}')
    # Decode image data
    logger.info('input_fn: Decode image data')
    json_data = json.loads(request_body)
    image_encoded = json_data['image']
    image_raw = base64.b64decode(image_encoded)
    # Load image from decoded bytestring
    image_bytes = np.frombuffer(image_raw, np.uint8)
    image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Transform image
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)[None].to(device)
    return image

def predict_fn(input_data, model):
    logger.info('predict_fn: Feeding to the model')
    with torch.inference_mode():
        logits = model(input_data)
        preds = torch.sigmoid(logits).cpu().item()
    return preds

def output_fn(prediction, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError(f'Unsupported request content type: {content_type}')
    logger.info('output_fn: Processing predictions')
    if prediction >= 0.5:
        content = dict(pred='Male', confidence=prediction)
    else:
        content = dict(pred='Female', confidence=1-prediction)
    return json.dumps(content)
