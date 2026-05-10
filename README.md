---
title: Construction Detection API
emoji: 🏗️
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: true
---

# Construction Detection API

Detects construction elements (beam, column, door, floor, stairs, wall, window) and measures their dimensions.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/detect` | POST | Upload image → get detections |
| `/calibrate` | POST | Set reference for real-world units |

## Usage

```python
import requests

r = requests.post(
    "https://dipangshuborah-construction-detection-api.hf.space/detect",
    files={"file": open("image.jpg", "rb")}
)
print(r.json())
```
