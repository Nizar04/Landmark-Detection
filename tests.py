# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_invalid_file():
    files = {"file": ("test.txt", "content", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

def test_prediction_valid_image():
    image_path = Path("tests/test_data/test_image.jpg")
    with open(image_path, "rb") as f:
        files = {"file": ("test_image.jpg", f, "image/jpeg")}
        response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) > 0

# tests/test_model.py
import pytest
import tensorflow as tf
import numpy as np
from app.model import LandmarkModel
from app.config import Config

def test_model_architecture():
    model = LandmarkModel()
    model.build()
    
    assert isinstance(model.model, tf.keras.Model)
    assert model.model.output_shape[-1] == Config.NUM_CLASSES

def test_model_prediction():
    model = LandmarkModel()
    model.build()
    
    dummy_input = np.random.rand(1, *Config.IMG_SIZE, 3)
    prediction = model.model.predict(dummy_input)
    
    assert prediction.shape == (1, Config.NUM_CLASSES)
    assert np.isclose(np.sum(prediction), 1.0)

# tests/conftest.py
import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_image():
    return np.random.rand(*Config.IMG_SIZE, 3)

@pytest.fixture
def test_model():
    model = LandmarkModel()
    model.build()
    return model
