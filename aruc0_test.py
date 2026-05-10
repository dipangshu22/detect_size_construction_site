import requests, base64

API_URL    = "https://newtechdevng-construction-detection-api.hf.space"
IMAGE_PATH = r"test3.jpg"

with open(IMAGE_PATH, "rb") as f:
    response = requests.post(
        f"{API_URL}/detect",
        files={"file": f},
        data={"marker_size_cm": 10.0, "confidence": 0.2}
    )

result = response.json()

print(f"✅ Success:     {result['success']}")
print(f"📐 Calibrated: {result['calibrated']}")
print(f"📏 Pixels/cm:  {result['pixels_per_cm']}")
print(f"⏱️  Inference:  {result['inference_time_s']}s")
print(f"🔍 Detections: {result['total']}\n")

for det in result["detections"]:
    if result["calibrated"]:
        print(f"  → {det['class']:<10} | conf: {det['confidence']:.2f} | "
              f"W: {det['width_cm']}cm | H: {det['height_cm']}cm")
    else:
        print(f"  → {det['class']:<10} | conf: {det['confidence']:.2f} | "
              f"W: {det['width_px']}px | H: {det['height_px']}px")

# Save annotated image
with open("result_aruco.jpg", "wb") as f:
    f.write(base64.b64decode(result["image_base64"]))
print("\n📸 Saved result_aruco.jpg — open it to see bounding boxes!")