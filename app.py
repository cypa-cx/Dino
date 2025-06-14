import os
import io
import base64
import gc
import torch
import requests
import numpy as np
from PIL import Image, ExifTags
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoImageProcessor
from rembg import remove, new_session
import uvicorn

# ========================================================================================
# KONFIGURACJA I MODELE GLOBALNE
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
# FUNKCJE POMOCNICZE
# ========================================================================================

def fix_image_orientation(image: Image.Image) -> Image.Image:
    """Naprawia orientację obrazu na podstawie EXIF"""
    try:
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            
            # Znajdź tag orientacji
            orientation_key = None
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_key = tag
                    break
            
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                
                # Mapowanie orientacji na rotację
                rotations = {3: 180, 6: 270, 8: 90}
                
                if orientation in rotations:
                    image = image.rotate(rotations[orientation], expand=True)
                    print(f"🔄 Obrócono obraz o {rotations[orientation]}°")
        
        return image
        
    except Exception as e:
        print(f"⚠️ Błąd naprawy orientacji: {e}")
        return image

def cleanup_memory():
    """Agresywne czyszczenie pamięci"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def log_memory_usage(stage: str):
    """Loguje użycie pamięci GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"💾 {stage}: {allocated:.2f} GB GPU")

# ========================================================================================
# INICJALIZACJA MODELI
# ========================================================================================

def initialize_models():
    """Inicjalizacja modeli przy starcie serwera"""
    global dinov2_model, dinov2_processor, rembg_session
    
    print("🔄 Ładowanie modeli...")
    
    # DINOv2-Large
    print("📥 Ładowanie DINOv2-Large...")
    dinov2_processor = AutoImageProcessor.from_pretrained(
        'facebook/dinov2-large',
        cache_dir='/tmp/hf_cache'
    )
    dinov2_model = AutoModel.from_pretrained(
        'facebook/dinov2-large',
        cache_dir='/tmp/hf_cache'
    )
    dinov2_model.eval()
    
    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_model = dinov2_model.to(device)
    print(f"🎯 DINOv2 załadowany na: {device}")
    
    # Rembg
    print("🖼️ Ładowanie rembg U2Net...")
    rembg_session = new_session('u2net')
    
    print("✅ Wszystkie modele załadowane!")

# ========================================================================================
# FUNKCJE PRZETWARZANIA OBRAZÓW
# ========================================================================================

def download_image(url: str) -> Image.Image:
    """Pobiera obraz z URL - zachowuje oryginalne rozmiary"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"📏 Pobrany obraz: {image.size}")
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie można pobrać obrazu: {str(e)}")

def remove_background(image: Image.Image) -> Image.Image:
    """Usuwa tło za pomocą rembg"""
    img_bytes = None
    img_bytes_data = None
    output = None
    
    try:
        # PIL -> bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes_data = img_bytes.getvalue()
        
        # Zamknij i wyczyść buffer
        img_bytes.close()
        del img_bytes
        img_bytes = None
        
        # Usuń tło
        output = remove(img_bytes_data, session=rembg_session)
        
        # Wyczyść input
        del img_bytes_data
        img_bytes_data = None
        
        # bytes -> PIL
        result_image = Image.open(io.BytesIO(output)).convert('RGBA')
        
        # Wyczyść output
        del output
        output = None
        
        print(f"🖼️ Tło usunięte: {result_image.size}")
        return result_image
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd usuwania tła: {str(e)}")
    finally:
        # Cleanup w przypadku błędu
        for var in [img_bytes, img_bytes_data, output]:
            if var is not None:
                del var
        gc.collect()

def crop_to_content(image: Image.Image) -> Image.Image:
    """Przycina obraz do granic nieprzezroczystych pikseli"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        bbox = image.getbbox()
        
        if bbox is None:
            print("⚠️ Nie znaleziono treści do przycięcia")
            return image
        
        cropped = image.crop(bbox)
        print(f"✂️ Przycięto z {image.size} do {cropped.size}")
        return cropped
        
    except Exception as e:
        print(f"❌ Błąd przycinania: {e}")
        return image

def generate_embedding(image: Image.Image) -> list[float]:
    """Generuje embedding za pomocą DINOv2"""
    inputs = None
    outputs = None
    embedding = None
    embedding_cpu = None
    
    try:
        # Konwersja na RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"🧠 Generowanie embeddingu dla: {image.size}")
        log_memory_usage("Przed embedding")
        
        # Preprocessing
        inputs = dinov2_processor(images=image, return_tensors="pt")
        device = next(dinov2_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = dinov2_model(**inputs)
            embedding = outputs.last_hidden_state[0, 0]  # [CLS] token
            embedding_cpu = embedding.cpu()
        
        # Konwersja do listy
        embedding_list = embedding_cpu.numpy().tolist()
        
        print(f"✅ Embedding: {len(embedding_list)} wymiarów")
        return embedding_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd embeddingu: {str(e)}")
    finally:
        # Cleanup wszystkich tensorów
        for var in [inputs, outputs, embedding, embedding_cpu]:
            if var is not None:
                del var
        
        cleanup_memory()
        log_memory_usage("Po embedding")

def image_to_base64(image: Image.Image) -> str:
    """Konwertuje obraz PIL na base64"""
    buffer = None
    img_bytes = None
    
    try:
        buffer = io.BytesIO()
        
        # Wybierz format na podstawie trybu
        if image.mode == 'RGBA':
            image.save(buffer, format='PNG')
        else:
            image.save(buffer, format='JPEG', quality=95)
        
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        print(f"📦 Base64 length: {len(img_base64)}")
        return img_base64
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd base64: {str(e)}")
    finally:
        # Cleanup
        if buffer:
            buffer.close()
        for var in [buffer, img_bytes]:
            if var is not None:
                del var
        gc.collect()

# ========================================================================================
# ENDPOINTS
# ========================================================================================

@app.on_event("startup")
async def startup_event():
    """Inicjalizacja przy starcie"""
    initialize_models()

@app.get("/")
async def root():
    """Endpoint główny"""
    return {
        "status": "healthy",
        "message": "DINO Embedding API is running! 🦕✨",
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
    """Health check z informacjami o pamięci"""
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "free_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3, 3)
        }
    
    return {
        "status": "ok",
        "models_loaded": dinov2_model is not None and rembg_session is not None,
        "gpu_available": torch.cuda.is_available(),
        "memory_info": memory_info
    }

@app.get("/stats")
async def get_stats():
    """Statystyki API"""
    global request_counter
    
    stats = {
        "total_requests": request_counter,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        stats["gpu_memory"] = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "free_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3, 3)
        }
    
    return stats

@app.get("/memory/clear")
async def clear_memory():
    """Ręczne czyszczenie pamięci"""
    cleanup_memory()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    return {
        "status": "memory_cleared",
        "message": "GPU cache and Python garbage collector cleared"
    }

@app.post("/process", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    """
    Główny endpoint przetwarzania obrazów
    Zachowuje oryginalne rozmiary i agresywnie czyści pamięć
    """
    global request_counter
    request_counter += 1
    
    # Zmienne do cleanup
    original_image = None
    fixed_image = None
    no_bg_image = None
    cropped_image = None
    rgb_image = None
    
    try:
        print(f"\n🔄 REQUEST #{request_counter}: {request.image_url}")
        log_memory_usage("Start")
        
        # 1. Pobierz obraz
        print("📥 Pobieranie...")
        original_image = download_image(request.image_url)
        
        # 2. Napraw orientację
        print("🔄 Orientacja...")
        fixed_image = fix_image_orientation(original_image)
        
        # Cleanup oryginalnego jeśli się zmienił
        if fixed_image is not original_image:
            del original_image
            original_image = None
            gc.collect()
        
        # 3. Usuń tło
        print("🖼️ Usuwanie tła...")
        no_bg_image = remove_background(fixed_image)
        
        del fixed_image
        fixed_image = None
        gc.collect()
        
        # 4. Przytnij
        print("✂️ Przycinanie...")
        cropped_image = crop_to_content(no_bg_image)
        
        if cropped_image is not no_bg_image:
            del no_bg_image
            no_bg_image = None
            gc.collect()
        
        # 5. Embedding
        print("🧠 Embedding...")
        rgb_image = cropped_image.convert('RGB')
        embedding = generate_embedding(rgb_image)
        
        del rgb_image
        rgb_image = None
        gc.collect()
        
        # 6. Base64
        print("📦 Base64...")
        image_base64 = image_to_base64(cropped_image)
        
        del cropped_image
        cropped_image = None
        gc.collect()
        
        # Finalne czyszczenie
        cleanup_memory()
        log_memory_usage("Koniec")
        
        print(f"✅ REQUEST #{request_counter}: SUKCES")
        
        return ImageResponse(
            embedding=embedding,
            image_base64=image_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ REQUEST #{request_counter}: BŁĄD - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd przetwarzania: {str(e)}")
    finally:
        # KRYTYCZNE: Cleanup wszystkich zmiennych
        for var_name, var_obj in [
            ('original_image', original_image),
            ('fixed_image', fixed_image),
            ('no_bg_image', no_bg_image),
            ('cropped_image', cropped_image),
            ('rgb_image', rgb_image)
        ]:
            if var_obj is not None:
                del var_obj
        
        # Agresywne czyszczenie
        cleanup_memory()
        print(f"🧹 REQUEST #{request_counter}: Pamięć wyczyszczona")

# ========================================================================================
# URUCHOMIENIE
# ========================================================================================

if __name__ == "__main__":
    print("🚀 Uruchamianie DINO Embedding API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
