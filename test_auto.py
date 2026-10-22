"""Automated test suite for the Flask hand-sign digit recognition app."""

import re
from io import BytesIO
from unittest.mock import patch

import pytest

from app import app as flask_app
from model import preprocess_img

SAMPLE_IMAGE = "test_images/2/Sign 2 (97).jpeg"
ERROR_MSG = b"File cannot be processed."


@pytest.fixture(name="client")
def client_fixture():
    """Flask test client for sending requests to the app."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    """Home page: should load and contain the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"<form" in response.data


def test_valid_upload_shows_prediction(client):
    """Valid request: with the model mocked, the predicted digit appears on the result page."""
    with patch("app.preprocess_img"), patch("app.predict_result", return_value=7):
        response = client.post(
            "/prediction",
            data={"file": (BytesIO(b"irrelevant when mocked"), "digit.jpg")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert ERROR_MSG not in response.data
    assert re.search(rb">\s*7\s*<", response.data)


def test_missing_file_shows_error(client):
    """Missing file: a request with no file field shows the friendly error."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert ERROR_MSG in response.data


def test_invalid_file_shows_error(client):
    """Invalid input: a file that is not an image shows the friendly error, not a crash."""
    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"this is not an image"), "fake.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert ERROR_MSG in response.data


def test_preprocess_img_output():
    """Preprocessing: a real image becomes a normalised array of the model's input shape."""
    result = preprocess_img(SAMPLE_IMAGE)
    assert result.shape == (1, 224, 224, 3)
    assert result.min() >= 0.0 and result.max() <= 1.0
