# import requests
# import base64

# API_URL = "https://newtechdevng-construction-detection-api.hf.space"
# IMAGE_PATH = r"test2.jpeg"

# with open(IMAGE_PATH, "rb") as f:
#     response = requests.post(f"{API_URL}/detect", files={"file": f})

# result = response.json()

# print(f"\n✅ Detections found: {result['total']}")
# print(f"⏱️  Inference time: {result['inference_time_s']:.3f}s")
# print(f"📐 Calibrated: {result['calibrated']}")
# print()

# for det in result["detections"]:
#     print(f"  → {det['class']:<10} | confidence: {det['confidence']:.2f} | "
#           f"W: {det['width_px']}px | H: {det['height_px']}px")

# # Save annotated image
# with open("result.jpg", "wb") as f:
#     f.write(base64.b64decode(result["image_base64"]))
# print("\n📸 Annotated image saved as result.jpg")


import requests

API_URL = "https://newtechdevng-construction-detection-api.hf.space"
IMAGE_PATH = r"test2.jpeg"

with open(IMAGE_PATH, "rb") as f:
    response = requests.post(
        f"{API_URL}/detect",
        files={"file": f},
        data={
            "marker_size_cm": 10.0,
            "confidence": 0.15   # ← adjust here
        }
    )

result = response.json()
print(f"📐 Calibrated: {result['calibrated']}")
print(f"✅ Detections: {result['total']}\n")

for det in result["detections"]:
    print(f"  → {det['class']:<10} | confidence: {det['confidence']:.2f} | "
          f"W: {det['width_cm']}cm | H: {det['height_cm']}cm")