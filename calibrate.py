import requests

API_URL = "https://newtechdevng-construction-detection-api.hf.space"
IMAGE_PATH = r"test2.jpeg"

with open(IMAGE_PATH, "rb") as f:
    response = requests.post(
        f"{API_URL}/calibrate",
        files={"file": f},
        data={
            "x1": 701,
            "y1": 237,
            "x2": 742,
            "y2": 439,
            "real_width_cm": 60.96,
            "real_height_cm": 335.28
        }
    )

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")