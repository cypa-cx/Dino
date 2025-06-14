import os
import io
import base64
import gc
import torch
import requests
from PIL import Image, ExifTags
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoImageProcessor
from rembg import remove, new_session
import uvicorn

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

app = FastAPI(title="DINO Embedding API", version="1.0.0")

# Modele globalne
dinov2_model = None
dinov2_processor = None
rembg_session = None

# Statystyki
request_counter = 0

# ========================================================================================
# MODELE DANYCH
# ========================================================================================

class ImageRequest(BaseModel):
    image_url: str

class ImageResponse(BaseModel):
    embedding: list[float]
    image_base64: str

# ========================================================================================
# SMART MEMORY MANAGEMENT
# ========================================================================================

def get_gpu_memory_usage() -> float:
    """Zwraca zużycie pamięci GPU w GB"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3

def should_cleanup() -> bool:
    """Cleanup TYLKO gdy faktycznie potrzeba"""
    if torch.cuda.is_available():
        memory_gb = get_gpu_memory_usage()
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_ratio = memory_gb / total_gb
        
        # Cleanup tylko gdy ponad 85% pamięci zajęte
        if usage_ratio > 0.85:
            print(f"🧹 Memory cleanup needed: {memory_gb:.2f}/{total_gb:.2f} GB ({usage_ratio*100:.1f}%)")
            return True
    
    return False

def efficient_cleanup():
    """Wydajne czyszczenie - minimum overhead"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ========================================================================================
# OPTIMIZED FUNCTIONS
# ========================================================================================

def fix_image_orientation(image: Image.Image) -> Image.Image:
    """Szybka naprawa orientacji - tylko essential cases"""
    try:
        exif = getattr(image, '_getexif', lambda: None)()
        if exif and 274 in exif:  # 274 = Orientation tag
            orientation = exif[274]
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                return image.rotate(rotations[orientation], expand=True)
    except:
        pass  # Ignore errors - nie warto crashować przez EXIF
    return image

def download_image(url: str) -> Image.Image:
    """Szybkie pobieranie z timeout optimization"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; DinoAPI/1.0)'}
        response = requests.get(url, headers=headers, timeout=15)  # Krótszy timeout
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot download image: {str(e)}")

def remove_background(image: Image.Image) -> Image.Image:
    """rembg z minimalnym overhead"""
    try:
        # Konwersja do bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        img_buffer.close()
        
        # rembg processing
        output = remove(img_data, session=rembg_session)
        result = Image.open(io.BytesIO(output)).convert('RGBA')
        
        # Cleanup tylko dużych objektów
        del img_data, output
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

def crop_to_content(image: Image.Image) -> Image.Image:
    """Szybkie przycinanie"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    bbox = image.getbbox()
    return image.crop(bbox) if bbox else image

def generate_embedding(image: Image.Image) -> list[float]:
    """Optimized embedding generation"""
    try:
        # RGB conversion
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocessing
        inputs = dinov2_processor(images=image, return_tensors="pt")
        device = next(dinov2_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = dinov2_model(**inputs)
            # Immediate CPU transfer
            embedding = outputs.last_hidden_state[0, 0].cpu().numpy().tolist()
        
        # Cleanup tensorów (bez gc.collect - za wolne)
        del inputs, outputs
        
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

def image_to_base64(image: Image.Image) -> str:
    """Konwersja do base64 - zachowuje original quality"""
    buffer = io.BytesIO()
    
    # Format jak w oryginalnej wersji
    if image.mode == 'RGBA':
        image.save(buffer, format='PNG')
    else:
        image.save(buffer, format='JPEG', quality=95)  # Original quality
    
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    
    return b64

# ========================================================================================
# INITIALIZATION
# ========================================================================================

def initialize_models():
    """Inicjalizacja modeli"""
    global dinov2_model, dinov2_processor, rembg_session
    
    print("🔄 Loading models...")
    
    # DINOv2-Large
    dinov2_processor = AutoImageProcessor.from_pretrained(
        'facebook/dinov2-large',
        cache_dir='/tmp/hf_cache'
    )
    dinov2_model = AutoModel.from_pretrained(
        'facebook/dinov2-large', 
        cache_dir='/tmp/hf_cache'
    ).eval()
    
    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_model = dinov2_model.to(device)
    
    # Optimize model for inference (bez JIT - nie wszystkie modele to wspierają)
    if torch.cuda.is_available():
        # Ustaw model w trybie inference
        dinov2_model.eval()
        for param in dinov2_model.parameters():
            param.requires_grad = False
    
    # rembg
    rembg_session = new_session('u2net')
    
    print("✅ Models loaded and optimized!")
    if torch.cuda.is_available():
        memory_gb = get_gpu_memory_usage()
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU Memory: {memory_gb:.2f}/{total_gb:.2f} GB ({memory_gb/total_gb*100:.1f}%)")

# ========================================================================================
# ENDPOINTS
# ========================================================================================

@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "DINO Embedding API - Optimized & Stable 🦕⚡",
        "gpu_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "model_info": {
            "dinov2": "facebook/dinov2-large",
            "rembg": "u2net",
            "embedding_size": 1536
        }
    }

@app.get("/health")
async def health_check():
    """Health check z info o pamięci"""
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
        "models_loaded": dinov2_model is not None and rembg_session is not None,
        "gpu_available": torch.cuda.is_available(),
        "memory_info": memory_info,
        "requests_processed": request_counter
    }

@app.get("/stats")
async def get_stats():
    """Szczegółowe statystyki"""
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
    """Ręczne czyszczenie pamięci"""
    efficient_cleanup()
    memory_gb = get_gpu_memory_usage()
    return {
        "status": "cleanup_completed", 
        "gpu_memory_gb": round(memory_gb, 2)
    }

@app.post("/process", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    """
    Główny endpoint - zoptymalizowany dla szybkości i stabilności
    """
    global request_counter
    request_counter += 1
    
    try:
        print(f"🔄 Request #{request_counter}: Processing...")
        
        # Smart cleanup - TYLKO gdy pamięć się zapełnia
        if should_cleanup():
            efficient_cleanup()
            print(f"🧹 Memory cleanup performed after request #{request_counter}")
        
        # Processing pipeline
        image = download_image(request.image_url)
        image = fix_image_orientation(image)
        image = remove_background(image)
        image = crop_to_content(image)
        
        # Embedding generation
        rgb_image = image.convert('RGB')
        embedding = generate_embedding(rgb_image)
        
        # Base64 conversion
        image_base64 = image_to_base64(image)
        
        print(f"✅ Request #{request_counter}: Completed")
        
        return ImageResponse(
            embedding=embedding,
            image_base64=image_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Request #{request_counter}: Error - {str(e)}")
        # Cleanup po błędzie
        efficient_cleanup()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ========================================================================================
# STARTUP
# ========================================================================================

if __name__ == "__main__":
    print("🚀 Starting DINO Embedding API - Optimized Version")
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=7860,
        log_level="info"
    )
import numpy as np

def alpha_threshold_bbox(image, min_alpha=30):
    alpha = np.array(image)[:, :, 3]
    rows = np.any(alpha >= min_alpha, axis=1)
    cols = np.any(alpha >= min_alpha, axis=0)
    if not np.any(rows) or not np.any(cols):
        return image.getbbox()  # fallback
    top = np.where(rows)[0][0]
    bottom = np.where(rows)[0][-1] + 1
    left = np.where(cols)[0][0]
    right = np.where(cols)[0][-1] + 1
    return (left, top, right, bottom)


