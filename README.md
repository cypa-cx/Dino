# 🦕⚡ DINO + VORTEX Embedding API

Advanced FastAPI service for generating high-quality embeddings from images using **DINOv2** and **VORTEX** models with intelligent background removal, advanced filtering, and memory optimization.

## 🎯 Key Features

- **Dual Embedding Models**: DINOv2-Large (1024 dims) + VORTEX-BeiTv2 (1024 dims)
- **Model Selection**: Choose between general-purpose (DINOv2) or texture-optimized (VORTEX) embeddings
- **Advanced Image Filters**: High-pass filtering and Canny edge detection
- **Flexible Processing**: Toggle background removal and content cropping
- **Smart Background Removal** using rembg with U2Net
- **Intelligent Memory Management** - handles 1000+ requests without crashes
- **GPU Acceleration** with CUDA optimization
- **Production-Ready** with comprehensive health monitoring
- **Optimized Processing Pipeline** with smart cropping and orientation fix

## 🚀 Available Models

### DINOv2-Large
- **Dimensions**: 1024
- **Architecture**: Vision Transformer (facebook/dinov2-large)
- **Strengths**: General-purpose visual features, semantic understanding
- **Best For**: Object recognition, scene understanding, product similarity
- **Model Parameter**: `"dinov2"`

### VORTEX (BeiTv2-Large)
- **Dimensions**: 1024
- **Architecture**: BeiT v2 with texture-optimized training
- **Strengths**: Texture analysis, fine-grained visual patterns, material recognition
- **Best For**: Surface analysis, fabric classification, texture similarity
- **Model Parameter**: `"vortex"`

## 🎨 Advanced Image Processing Options

### Background Processing
- **`crop: true`** (default) - Remove background + crop to content
- **`crop: false`** - Keep original image with background

### Image Enhancement Filters
- **`high_pass: true`** - Apply high-pass filter for detail enhancement
- **`canny: true`** - Apply Canny edge detection for edge analysis

## 📡 API Endpoints

### Core Processing

#### `POST /process` - Universal Image Processing Endpoint
**Request Parameters:**
```json
{
  "image_url": "https://example.com/image.jpg",
  "model": "dinov2",           // "dinov2" | "vortex" (default: dinov2)
  "crop": true,                // Remove background & crop (default: true)
  "high_pass": false,          // High-pass filter for details (default: false)
  "canny": false               // Canny edge detection (default: false)
}
```

**Response:**
```json
{
  "embedding": [1024 float values],
  "image_base64": "base64 encoded processed image",
  "model_used": "DINOv2-Large",
  "embedding_dimensions": 1024,
  "background_removed": true,
  "filters_applied": ["background_removal", "crop"]
}
```

### Model Information

#### `GET /models` - Available Models & Usage Examples
Returns detailed information about available models and parameter combinations.

### Legacy Endpoints (Deprecated)
- `POST /process_dinov2` - DINOv2 only (use `/process` with `model: "dinov2"`)
- `POST /process_vortex` - VORTEX only (use `/process` with `model: "vortex"`)

### Monitoring & Health

- **`GET /`** - API status and model availability
- **`GET /health`** - Detailed health check with memory usage
- **`POST /memory/cleanup`** - Manual memory cleanup

## 🔧 RunPod Deployment

### Quick Deploy Template

```bash
bash -c 'cd /workspace && rm -rf Dino && git clone https://github.com/cypa-cx/Dino.git && cd Dino && git clone https://github.com/scabini/VORTEX.git && pip install --no-cache-dir -r requirements.txt && pip install timm==1.0.11 einops==0.4.1 scikit-learn && python app.py'
```

### RunPod Template Settings
- **Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **GPU**: RTX 4090, A5000, or similar (8GB+ VRAM recommended)
- **Storage**: 25GB+ for models and dependencies and dependencies

## 🏠 Local Development

```bash
# Clone repository
git clone https://github.com/cypa-cx/Dino.git
cd Dino

# Clone VORTEX
git clone https://github.com/scabini/VORTEX.git

# Install dependencies
pip install -r requirements.txt
pip install timm==1.0.11 einops==0.4.1 scikit-learn

# Start server
python app.py
```

API will be available at `http://localhost:7860`

## 🧪 Usage Examples

### Python Client

```python
import requests

# Basic DINOv2 embedding (default settings)
response = requests.post(
    "http://your-api:7860/process",
    json={"image_url": "https://example.com/product.jpg"}
)

# VORTEX texture analysis
response_texture = requests.post(
    "http://your-api:7860/process",
    json={
        "image_url": "https://example.com/fabric.jpg",
        "model": "vortex"
    }
)

# Edge detection + DINOv2 (useful for drawings/sketches)
response_edges = requests.post(
    "http://your-api:7860/process",
    json={
        "image_url": "https://example.com/drawing.jpg",
        "model": "dinov2",
        "canny": True,
        "crop": False  # Keep background for edge analysis
    }
)

# High-pass filter + VORTEX (texture detail enhancement)
response_detail = requests.post(
    "http://your-api:7860/process",
    json={
        "image_url": "https://example.com/material.jpg",
        "model": "vortex",
        "high_pass": True
    }
)

# Compare embeddings
dino_embedding = response.json()["embedding"]
vortex_embedding = response_texture.json()["embedding"]

from scipy.spatial.distance import cosine
similarity = 1 - cosine(dino_embedding, vortex_embedding)
print(f"Cross-model similarity: {similarity:.3f}")
```

### cURL Examples

```bash
# Basic DINOv2 processing
curl -X POST "http://your-api:7860/process" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'

# VORTEX texture analysis with high-pass filter
curl -X POST "http://your-api:7860/process" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/texture.jpg",
    "model": "vortex",
    "high_pass": true
  }'

# Edge detection analysis
curl -X POST "http://your-api:7860/process" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/drawing.jpg",
    "canny": true,
    "crop": false
  }'

# Get available models and usage examples
curl "http://your-api:7860/models"

# Health check
curl "http://your-api:7860/health"
```

## 🎨 Image Processing Pipeline

1. **Download & Validation** - Robust image fetching with timeout handling
2. **Orientation Fix** - Automatic EXIF orientation correction
3. **Background Processing** - Optional U2Net-based background removal
4. **Content Cropping** - Smart alpha-threshold based cropping (if enabled)
5. **Image Filtering** - Optional high-pass or Canny edge detection filters
6. **Model-Specific Preprocessing** - Tailored preprocessing for selected model
7. **Embedding Generation** - Feature extraction using chosen model
8. **Memory Optimization** - Intelligent cleanup and GPU memory management

## 🔧 Technical Specifications

### System Requirements
- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU acceleration)
- **GPU Memory**: 8GB+ recommended
- **RAM**: 16GB+ recommended
- **Storage**: 25GB+ for models

### Dependencies
- **Core**: PyTorch 2.0+, FastAPI, Transformers
- **Vision**: Pillow, OpenCV, rembg
- **VORTEX**: timm, einops, scikit-learn
- **Deployment**: Uvicorn, pydantic, requests

### Performance Optimization
- **Smart Memory Management**: Cleanup triggered only when needed (>85% GPU usage)
- **Model Optimization**: Inference-only mode with disabled gradients
- **Efficient Processing**: Minimal tensor operations and immediate cleanup
- **Batch Processing**: Optimized for high-throughput scenarios

## 📊 Model Comparison

| Feature | DINOv2-Large | VORTEX-BeiTv2 |
|---------|--------------|---------------|
| **Dimensions** | 1024 | 1024 |
| **Training Data** | ImageNet-22k | ImageNet-1k + Texture |
| **Architecture** | Vision Transformer | BeiT v2 |
| **Best Use Case** | General similarity | Texture analysis |
| **Processing Speed** | ~2-3s per image | ~2-3s per image |
| **Memory Usage** | ~4GB VRAM | ~3GB VRAM |
| **Background Removal** | Yes (U2Net) | Yes (U2Net) |
| **Smart Cropping** | Yes (Alpha threshold) | Yes (Alpha threshold) |

## 🔍 Use Case Recommendations

### Use DINOv2 (`model: "dinov2"`) for:
- **Product Similarity**: E-commerce recommendation systems
- **General Classification**: Object detection, scene understanding
- **Semantic Search**: Content-based image retrieval
- **Object Recognition**: General-purpose visual understanding

### Use VORTEX (`model: "vortex"`) for:
- **Material Classification**: Identifying fabrics, surfaces, textures
- **Texture Analysis**: Fine-grained pattern recognition
- **Fashion/Textile**: Clothing and fabric similarity
- **Surface Analysis**: Material quality assessment

### Advanced Processing Combinations:

**High-Pass Filter** (`high_pass: true`):
- Enhances fine details and textures
- Ideal for: Material analysis, texture classification
- Best with: VORTEX model for texture tasks

**Canny Edge Detection** (`canny: true`):
- Extracts edge and contour information
- Ideal for: Drawings, sketches, architectural images
- Best with: DINOv2 for shape-based similarity

**No Background Removal** (`crop: false`):
- Preserves context and background information
- Ideal for: Scene understanding, contextual analysis

## 🛠️ Troubleshooting

### Common Issues

**VORTEX Import Error**
```
Solution: Ensure VORTEX repository is cloned and dependencies installed
git clone https://github.com/scabini/VORTEX.git
pip install timm==1.0.11 einops==0.4.1 scikit-learn
```

**Model Selection Error**
```
Solution: Use valid model names: "dinov2" or "vortex"
Check /models endpoint for available models and examples
```

**GPU Out of Memory**
```
Solution: Monitor memory usage via /health endpoint
Use manual cleanup: POST /memory/cleanup
Reduce concurrent requests if needed
```

**Filter Processing Failures**
```
Solution: Ensure OpenCV is properly installed
Some filters may fail gracefully and return original image
Check response.filters_applied to verify which filters were applied
```

### Performance Tuning

For high-throughput scenarios:
- Use GPU instances with 16GB+ VRAM
- Monitor `/health` endpoint for memory usage  
- Enable automatic cleanup (triggers at >85% usage)
- Consider load balancing for >100 concurrent requests
- Use appropriate model for task (DINOv2 for general, VORTEX for textures)

## 📈 Monitoring & Health

### Health Check Response
```json
{
  "status": "ok",
  "requests_processed": 1247,
  "models": {
    "dinov2": true,
    "vortex": true,
    "rembg": true
  },
  "gpu": true,
  "gpu_memory": {
    "allocated_gb": 6.2,
    "total_gb": 24.0,
    "usage_percent": 25.8
  },
  "vortex_details": {
    "available": true,
    "backbone": "beitv2_large_patch16_224.in1k_ft_in22k_in1k",
    "input_size": 224
  }
}
```

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns for memory optimization
- Both DINOv2 and VORTEX models remain supported
- New filters are implemented with graceful error handling
- API compatibility is maintained for existing parameters
- Performance optimizations are preserved

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

### Third-Party Licenses

- **DINOv2**: Apache 2.0 License - https://github.com/facebookresearch/dinov2
- **rembg**: MIT License - https://github.com/danielgatis/rembg

## 🙏 Acknowledgments

- **DINOv2**: Visual features by Meta AI Research
- **rembg**: Background removal by @danielgatis

---

**🚀 Ready to deploy? Use the RunPod template above for instant setup!**
