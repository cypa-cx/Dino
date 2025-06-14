# 🦕 DINO Embedding API

FastAPI service for generating embeddings from images using DINOv2 model with background removal.

## Features

- **DINOv2-Large** embeddings (1536 dimensions)
- **Background removal** using rembg
- **Memory management** - handles 1000+ requests without crashes
- **GPU acceleration** with CUDA support
- **Health monitoring** endpoints

## API Endpoints

### `POST /process`
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

Response:
```json
{
  "embedding": [1536 float values],
  "image_base64": "base64 encoded processed image"
}
```

### `GET /health`
Memory and GPU status

### `GET /stats`
Request statistics and memory usage

### `GET /memory/clear`
Manual memory cleanup

## RunPod Deployment

Use this Docker Command in RunPod template:

```bash
bash -c 'cd /workspace && git clone https://github.com/cypa-cx/Dino.git && cd Dino && pip install -r requirements.txt && python app.py'
```

## Local Development

```bash
git clone https://github.com/cypa-cx/Dino.git
cd Dino
pip install -r requirements.txt
python app.py
```

API will be available at `http://localhost:7860`

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for optimal performance
