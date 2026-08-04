"""Tests for image_utils.load_images."""
import io
import os

import pytest
from PIL import Image

from app.image_utils import load_images


@pytest.fixture
def sample_jpeg(tmp_path):
    path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (16, 16), color=(200, 10, 10))
    img.save(path, format="JPEG")
    return str(path)


@pytest.fixture
def sample_png(tmp_path):
    path = tmp_path / "sample.png"
    img = Image.new("RGBA", (8, 8), color=(0, 255, 0, 255))
    img.save(path, format="PNG")
    return str(path)


def test_load_jpeg_returns_data_url(sample_jpeg):
    urls = load_images([sample_jpeg])
    assert len(urls) == 1
    assert urls[0].startswith("data:image/jpeg;base64,")


def test_missing_file_is_skipped(tmp_path, sample_jpeg):
    urls = load_images([sample_jpeg, str(tmp_path / "does_not_exist.jpg")])
    assert len(urls) == 1


def test_png_is_reencoded_to_jpeg(sample_png):
    urls = load_images([sample_png])
    assert len(urls) == 1
    assert urls[0].startswith("data:image/jpeg;base64,")


def test_empty_and_none_inputs():
    assert load_images([]) == []
    assert load_images(None) == []
    assert load_images(["", None]) == []


def test_corrupt_file_is_skipped(tmp_path, sample_jpeg):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    urls = load_images([sample_jpeg, str(bad)])
    assert len(urls) == 1

