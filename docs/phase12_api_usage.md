# GaitGuard AI — Production API Usage & Development Guide

## 1. Local Server Launch Instructions

To launch the GaitGuard AI FastAPI production server locally:

```bash
# Set PYTHONPATH to project root
$env:PYTHONPATH="."

# Launch uvicorn server on port 8000
py -3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation will be accessible at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 2. Tested Example Requests

### Health Check (`cURL`):
```bash
curl -X GET http://localhost:8000/health
```

### Version Check (`cURL`):
```bash
curl -X GET http://localhost:8000/version
```

### Video Screening Request (`cURL`):
```bash
curl -X POST http://localhost:8000/api/v1/screen \
  -F "video=@sample_cow_walk.mp4" \
  -F "animal_id=cow_99" \
  -F "session_id=sess_20261008"
```

### Video Screening Request (`Python requests`):
```python
import requests

url = "http://localhost:8000/api/v1/screen"
files = {"video": ("cow_walk.mp4", open("sample_cow_walk.mp4", "rb"), "video/mp4")}
data = {"animal_id": "cow_99", "session_id": "sess_20261008"}

response = requests.post(url, files=files, data=data)
print(response.json())
```
