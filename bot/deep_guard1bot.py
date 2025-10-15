import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import io
import logging
import cv2
import numpy as np
import mediapipe as mp

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 300
RESIZE_SIZE = 380

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  
    min_detection_confidence=0.5
)

def load_model(model_path='bot/best_model2.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=num_ftrs, out_features=2)
    )
    
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device), 
            strict=True
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise
    
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

test_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🔍 Привет! Отправь мне фотографию лица, и я проверю ее на наличие признаков дипфейка."
    )

def detect_head_regions(image):
    image_np = np.array(image)
    results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    if not results.detections:
        return []
    
    crops = []
    height, width, _ = image_np.shape
    
    for detection in results.detections:
        box = detection.location_data.relative_bounding_box
        expand = 0.5
        xmin = max(0, int((box.xmin - box.width * expand) * width))
        ymin = max(0, int((box.ymin - box.height * expand) * height))
        xmax = min(width, int((box.xmin + box.width * (1 + expand)) * width))
        ymax = min(height, int((box.ymin + box.height * (1 + expand)) * height))
        
        crops.append(image.crop((xmin, ymin, xmax, ymax)))
    
    return crops

async def process_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')

        head_regions = detect_head_regions(image)
        if not head_regions:
            return "❌ Лица не обнаружены. Убедитесь, что лица хорошо видны."
            
        is_fake = False
        details = []
        FAKE_THRESHOLD = 50 
        
        for i, face in enumerate(head_regions, 1):
            tensor = test_transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                
            fake_prob = probs[0][0].item() * 100
            real_prob = probs[0][1].item() * 100
            
            face_status = "🚫 ДИПФЕЙК" if fake_prob > FAKE_THRESHOLD else "✅ НАСТОЯЩЕЕ"
            if fake_prob > FAKE_THRESHOLD:
                is_fake = True
                
            details.append(
                f"Лицо {i}:\n"
                f"{face_status}\n"
                f"Вероятность подделки: {fake_prob:.1f}%\n"
            )
        
        result = [
            "🔍 Результаты анализа:",
            f"Обнаружено лиц: {len(head_regions)}",
            *details,
            "\nИтоговый вердикт:",
            "🚨 ВНИМАНИЕ! Обнаружен дипфейк!" if is_fake 
            else "🟢 Фото выглядит аутентичным"
            
        ]
        
        return "\n".join(result)
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)[:200]}")
        return "❌ Ошибка обработки изображения. Пожалуйста, попробуйте другое фото."

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if update.message.photo[-1].file_size > 5*1024*1024:
            await update.message.reply_text("⚠️ Изображение слишком большое (макс. 5 МБ)")
            return
            
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        
        result = await process_image(image_bytes)
        await update.message.reply_text(result)
    except Exception as e:
        logger.error(f"Ошибка: {str(e)[:200]}")
        await update.message.reply_text("🚨 Произошла внутренняя ошибка. Попробуйте позже.")

def main() -> None:
    application = Application.builder().token("7978463391:AAHK5y9h2nxNvtZdkT_Lv9ceIXHaWuUGmIw").build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(
        filters.PHOTO & ~filters.FORWARDED, 
        handle_photo
    ))
    
    application.run_polling()

if __name__ == "__main__":
    main()