import os
import io
import base64
import gc
import torch
import requests
import numpy as np
import sys
from PIL import Image, ExifTags
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoImageProcessor
from rembg import remove, new_session
import uvicorn

# VORTEX Integration - Fixed Import Path
VORTEX_AVAILABLE = False
vortex_feature_extractor = None

def initialize_vortex():
    """Initialize VORTEX with proper error handling and working directory"""
    global VORTEX_AVAILABLE, vortex_feature_extractor
    
    try:
        # Save current working directory
        original_cwd = os.getcwd()
        vortex_path = os.path.join(original_cwd, 'VORTEX')
        
        if not os.path.exists(vortex_path):
            print(f"❌ VORTEX path not found: {vortex_path}")
            return False
        
        print(f"✅ VORTEX path found: {vortex_path}")
        
        # Add VORTEX to Python path
        sys.path.insert(0, vortex_path)
        
        # Change to VORTEX directory (needed for weight files)
        os.chdir(vortex_path)
        print(f"🔄 Changed working directory to: {os.getcwd()}")
        
        # Verify required files exist
        required_files = ['models.py', 'RAE_LCG_weights.pkl']
        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ Required file missing: {file}")
                os.chdir(original_cwd)  # Restore original directory
                return False
            print(f"✅ Found required file: {file}")
        
        # Try importing VORTEX
        from models import VORTEX
        print("✅ VORTEX module imported successfully")
        
        # Initialize VORTEX with BeiTv2-Large
        backbone = 'beitv2_large_patch16_224.in1k_ft_in22k_in1k'
        input_size = 224
        print(f"🔄 Initializing VORTEX with backbone: {backbone}")
        
        vortex_feature_extractor = VORTEX(backbone, input_size)
        
        # Restore original working directory
        os.chdir(original_cwd)
        print(f"🔄 Restored working directory to: {os.getcwd()}")
        
        VORTEX_AVAILABLE = True
        print("✅ VORTEX (BeiTv2-Large) loaded successfully!")
        return True
        
    except ImportError as e:
        print(f"⚠️ VORTEX import failed: {e}")
        # Restore working directory on error
        try:
            os.chdir(original_cwd)
        except:
            pass
        VORTEX_AVAILABLE = False
        return False
    except Exception as e:
        print(f"❌ VORTEX initialization failed: {e}")
        # Restore working directory on error
        try:
            os.chdir(original_cwd)
        except:
            pass
        VORTEX_AVAILABLE = False
        return False

# ========================================================================================
# CONFIGURATION
# ========================================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🔄 Starting application...")
    initialize_models()
    yield
    # Shutdown
    print("🔄 Shutting down application...")

app = FastAPI(
    title="DINO + VORTEX Embedding API", 
    version="2.0.0",
    description="DINOv2 + VORTEX texture analysis endpoints",
    lifespan=lifespan
)

# Global models
dinov2_model = None
dinov2_processor = None
rembg_session = None

# Statistics
request_counter = 0

# ========================================================================================
# DATA MODELS
# ========================================================================================

class ImageRequest(BaseModel):
    image_url: str

class ImageResponse(BaseModel):
    embedding: list[float]
    image_base64: str

class HealthResponse(BaseModel):
    status: str
    message: str
    models_available: dict
    gpu_info: dict

# ========================================================================================
# MEMORY MANAGEMENT
# ========================================================================================

def get_gpu_memory_usage() -> float:
    """Returns GPU memory usage in GB"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3

def should_cleanup() -> bool:
    """Cleanup only when really needed"""
    if torch.cuda.is_available():
        memory_gb = get_gpu_memory_usage()
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_ratio = memory_gb / total_gb
        
        if usage_ratio > 0.85:
            print(f"🧹 Memory cleanup needed: {memory_gb:.2f}/{total_gb:.2f} GB ({usage_ratio*100:.1f}%)")
            return True
    return False

def efficient_cleanup():
    """Efficient cleanup - minimum overhead"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ========================================================================================
# IMAGE PROCESSING FUNCTIONS
# ========================================================================================

def alpha_threshold_bbox(image, min_alpha=30):
    """Find bounding box based on pixels with alpha >= min_alpha"""
    alpha = np.array(image)[:, :, 3]
    rows = np.any(alpha >= min_alpha, axis=1)
    cols = np.any(alpha >= min_alpha, axis=0)
    if not np.any(rows) or not np.any(cols):
        return image.getbbox()
    top = np.where(rows)[0][0]
    bottom = np.where(rows)[0][-1] + 1
    left = np.where(cols)[0][0]
    right = np.where(cols)[0][-1] + 1
    return (left, top, right, bottom)

def fix_image_orientation(image: Image.Image) -> Image.Image:
    """Quick orientation fix"""
    try:
        exif = getattr(image, '_getexif', lambda: None)()
        if exif and 274 in exif:
            orientation = exif[274]
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                return image.rotate(rotations[orientation], expand=True)
    except:
        pass
    return image

def download_image(url: str) -> Image.Image:
    """Download image with timeout"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; DinoVortexAPI/2.0)'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot download image: {str(e)}")

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using rembg"""
    try:
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        img_buffer.close()
        
        output = remove(img_data, session=rembg_session)
        result = Image.open(io.BytesIO(output)).convert('RGBA')
        
        del img_data, output
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

def crop_to_content(image: Image.Image) -> Image.Image:
    """Crop to content"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    bbox = alpha_threshold_bbox(image, min_alpha=30)
    return image.crop(bbox) if bbox else image

def image_to_base64(image: Image.Image) -> str:
    """Convert to base64"""
    buffer = io.BytesIO()
    if image.mode == 'RGBA':
        image.save(buffer, format='PNG')
    else:
        image.save(buffer, format='JPEG', quality=95)
    
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return b64

# ========================================================================================
# MODEL INITIALIZATION
# ========================================================================================

def initialize_models():
    """Initialize both DINOv2 and VORTEX models"""
    global dinov2_model, dinov2_processor, rembg_session
    
    print("🔄 Loading models...")
    
    # DINOv2-Large
    try:
        dinov2_processor = AutoImageProcessor.from_pretrained(
            'facebook/dinov2-large',
            cache_dir='/tmp/hf_cache'
        )
        dinov2_model = AutoModel.from_pretrained(
            'facebook/dinov2-large', 
            cache_dir='/tmp/hf_cache'
        ).eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dinov2_model = dinov2_model.to(device)
        
        if torch.cuda.is_available():
            dinov2_model.eval()
            for param in dinov2_model.parameters():
                param.requires_grad = False
        
        print("✅ DINOv2-Large loaded successfully!")
    except Exception as e:
        print(f"❌ DINOv2 loading failed: {e}")
        dinov2_model = None
    
    # VORTEX Model
    initialize_vortex()
    
    # rembg
    try:
        rembg_session = new_session('u2net')
        print("✅ Background removal loaded successfully!")
    except Exception as e:
        print(f"❌ rembg loading failed: {e}")

# ========================================================================================
# DINOV2 EMBEDDING GENERATION
# ========================================================================================

def generate_dinov2_embedding(image: Image.Image) -> list[float]:
    """Generate DINOv2 embedding"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = dinov2_processor(images=image, return_tensors="pt")
        device = next(dinov2_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = dinov2_model(**inputs)
            embedding = outputs.last_hidden_state[0, 0].cpu().numpy().tolist()
        
        del inputs, outputs
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DINOv2 embedding failed: {str(e)}")

# ========================================================================================
# VORTEX EMBEDDING GENERATION  
# ========================================================================================

def generate_vortex_embedding(image: Image.Image) -> list[float]:
    """Generate VORTEX embedding"""
    try:
        from torchvision import transforms
        
        # VORTEX preprocessing (according to their documentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            # Use the global VORTEX feature extractor
            vortex_features = vortex_feature_extractor(image_tensor)
            embedding = vortex_features.squeeze().cpu().numpy().tolist()
        
        del image_tensor, vortex_features
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VORTEX embedding failed: {str(e)}")

# ========================================================================================
# ENDPOINTS
# ========================================================================================

# Usunięte - zastąpione przez lifespan

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check with model status"""
    gpu_info = {}
    if torch.cuda.is_available():
        allocated = get_gpu_memory_usage()
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "allocated_gb": round(allocated, 2),
            "total_gb": round(total, 2),
            "usage_percent": round((allocated/total)*100, 1)
        }
    else:
        gpu_info = {"available": False}
    
    return HealthResponse(
        status="healthy",
        message="DINO + VORTEX Embedding API 🦕⚡",
        models_available={
            "dinov2_large": dinov2_model is not None,
            "vortex_beitv2": VORTEX_AVAILABLE and vortex_feature_extractor is not None,
            "background_removal": rembg_session is not None
        },
        gpu_info=gpu_info
    )

@app.post("/process", response_model=ImageResponse)
async def process_dinov2(request: ImageRequest):
    """DINOv2 processing endpoint (1536 dimensions)"""
    global request_counter
    request_counter += 1
    
    if dinov2_model is None:
        raise HTTPException(status_code=503, detail="DINOv2 model not available")
    
    try:
        print(f"🔄 DINOv2 Request #{request_counter}: Processing...")
        
        # Smart cleanup
        if should_cleanup():
            efficient_cleanup()
            print(f"🧹 Memory cleanup performed after request #{request_counter}")
        
        # Image processing pipeline
        image = download_image(request.image_url)
        image = fix_image_orientation(image)
        image = remove_background(image)
        image = crop_to_content(image)
        
        # DINOv2 embedding
        rgb_image = image.convert('RGB')
        embedding = generate_dinov2_embedding(rgb_image)
        
        # Base64 conversion
        image_base64 = image_to_base64(image)
        
        print(f"✅ DINOv2 Request #{request_counter}: Completed ({len(embedding)} dims)")
        
        return ImageResponse(
            embedding=embedding,
            image_base64=image_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ DINOv2 Request #{request_counter}: Error - {str(e)}")
        efficient_cleanup()
        raise HTTPException(status_code=500, detail=f"DINOv2 processing failed: {str(e)}")

@app.post("/process_vortex", response_model=ImageResponse)
async def process_vortex(request: ImageRequest):
    """VORTEX processing endpoint (texture-optimized features)"""
    global request_counter
    request_counter += 1
    
    if not VORTEX_AVAILABLE:
        raise HTTPException(status_code=503, detail="VORTEX not available - check startup logs")
    
    if vortex_feature_extractor is None:
        raise HTTPException(status_code=503, detail="VORTEX model not initialized")
    
    try:
        print(f"🔄 VORTEX Request #{request_counter}: Processing...")
        
        # Smart cleanup
        if should_cleanup():
            efficient_cleanup()
        
        # Same image processing pipeline
        image = download_image(request.image_url)
        image = fix_image_orientation(image)
        image = remove_background(image)
        image = crop_to_content(image)
        
        # VORTEX embedding
        rgb_image = image.convert('RGB')
        embedding = generate_vortex_embedding(rgb_image)
        
        # Base64 conversion
        image_base64 = image_to_base64(image)
        
        print(f"✅ VORTEX Request #{request_counter}: Completed ({len(embedding)} dims)")
        
        return ImageResponse(
            embedding=embedding,
            image_base64=image_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ VORTEX Request #{request_counter}: Error - {str(e)}")
        efficient_cleanup()
        raise HTTPException(status_code=500, detail=f"VORTEX processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    memory_info = {}
    if torch.cuda.is_available():
        allocated = get_gpu_memory_usage()
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_info = {
            "allocated_gb": round(allocated, 2),
            "total_gb": round(total, 2), 
            "usage_percent": round((allocated/total)*100, 1),
            "free_gb": round(total - allocated, 2)
        }
    
    return {
        "status": "ok",
        "requests_processed": request_counter,
        "models": {
            "dinov2": dinov2_model is not None,
            "vortex": VORTEX_AVAILABLE and vortex_feature_extractor is not None,
            "rembg": rembg_session is not None
        },
        "gpu_available": torch.cuda.is_available(),
        "memory_info": memory_info,
        "vortex_details": {
            "available": VORTEX_AVAILABLE,
            "backbone": "beitv2_large_patch16_224.in1k_ft_in22k_in1k" if VORTEX_AVAILABLE else None,
            "input_size": 224 if VORTEX_AVAILABLE else None
        }
    }

@app.get("/stats")
async def get_stats():
    """Detailed statistics"""
    stats = {
        "total_requests": request_counter,
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        allocated = get_gpu_memory_usage()
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        stats["gpu_memory"] = {
            "allocated_gb": round(allocated, 2),
            "total_gb": round(total, 2),
            "usage_percent": round((allocated/total)*100, 1)
        }
    
    return stats

@app.post("/memory/cleanup")
async def manual_cleanup():
    """Manual memory cleanup"""
    efficient_cleanup()
    memory_gb = get_gpu_memory_usage()
    return {
        "status": "cleanup_completed", 
        "gpu_memory_gb": round(memory_gb, 2)
    }

# ========================================================================================
# STARTUP
# ========================================================================================

if __name__ == "__main__":
    print("🚀 Starting DINO + VORTEX Embedding API")
    print("""
    🦕 DINOv2: General-purpose visual embeddings (1536 dims)
    🌪️ VORTEX: Texture Analysis with Vision Transformers
    Copyright (c) 2025 scabini - Licensed under MIT License
    https://github.com/scabini/VORTEX
    """)
    
    # Find available port
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    # Try port 7860, if busy use random free port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', 7860))
            port = 7860
    except OSError:
        port = find_free_port()
        print(f"⚠️ Port 7860 busy, using port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
