#!/usr/bin/env python3
"""Quick API test to diagnose the issue"""
import requests
from pathlib import Path

API = "http://localhost:5000/predict"
TEST_IMAGE = Path("data/splits/test/PASS/00001.jpg")

# Try to find any test image
test_files = list(Path("data/splits/test").rglob("*.jpg"))
if test_files:
    TEST_IMAGE = test_files[0]

print(f"Testing with: {TEST_IMAGE}")
print(f"File exists: {TEST_IMAGE.exists()}")

if TEST_IMAGE.exists():
    with open(TEST_IMAGE, "rb") as f:
        files = {"image": f}
        response = requests.post(API, files=files)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
