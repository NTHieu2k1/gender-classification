import base64
import json
import numpy as np
import cv2 as cv
from streamlit.runtime.uploaded_file_manager import UploadedFile


def load_image(file: UploadedFile):
    image_raw = file.read()
    image_encoded = base64.b64encode(image_raw).decode()
    json_data = json.dumps(dict(image=image_encoded))
    image_bytes = np.frombuffer(image_raw, np.uint8)
    image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image, json_data
