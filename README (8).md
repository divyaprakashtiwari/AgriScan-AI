# AgriScan AI 🌱

**AI-Powered Crop Disease & Pest Detection System**

AgriScan AI lets any farmer send a photo of a diseased crop leaf to a Telegram bot and receive an instant AI diagnosis — disease name, confidence score, and treatment pointer — in both **English and Hindi**. No app installation, no login, no agricultural expert required.

---

## 🧠 How It Works

1. Farmer sends a leaf photo to the Telegram bot.
2. **n8n** receives the photo and forwards it to the prediction API.
3. **FastAPI** preprocesses the image and runs it through a trained **MobileNetV2** CNN.
4. The model returns the predicted disease + confidence score.
5. If confidence ≥ 60%, n8n sends back a formatted diagnosis (English + Hindi). If not, it asks for a clearer photo.

```
Farmer → Telegram Bot → n8n → FastAPI → MobileNetV2 Model → Prediction → n8n → Telegram Reply
```

---

## 🩺 Supported Crops & Diseases

15 classes across **Pepper (bell), Potato, and Tomato**:

- Pepper: Bacterial spot, Healthy
- Potato: Early blight, Late blight, Healthy
- Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | MobileNetV2 (transfer learning, PyTorch) |
| Dataset | PlantVillage (Kaggle) |
| Training | Google Colab (GPU) |
| Backend API | FastAPI + Uvicorn |
| Containerization | Docker / Docker Compose |
| Automation | n8n |
| User Interface | Telegram Bot API |
| Language Support | English + Hindi (curated mapping) |

---

## 📁 Project Structure

```
fastapi-app/
  main.py                  # FastAPI app: model loading + /predict endpoint
  crop_disease_model.pth   # Trained MobileNetV2 weights
  class_names.json         # English class labels
  class_names_hindi.json   # Hindi class label mapping
  Dockerfile
  requirements.txt
docker-compose.yml          # Runs FastAPI + n8n together
```

---

## 🚀 Setup & Run

### 1. Train the model (Google Colab)
- Dataset: PlantVillage subset (15 classes)
- Model: MobileNetV2, fine-tuned via transfer learning
- Epochs: 10 · Batch size: 32 · Optimizer: Adam (lr = 0.001)
- Output: `crop_disease_model.pth` + `class_names.json`

### 2. Deploy the API
```bash
docker compose up -d --build
```
This runs FastAPI (`localhost:8000`) and n8n (`localhost:5678`) together.

### 3. Connect Telegram
- Create a bot via [@BotFather](https://t.me/BotFather)
- Add the bot token as a credential in n8n
- Import the workflow: `Telegram Trigger → Get File → HTTP Request → Confidence Check → Format Reply → Send Reply`

### 4. Test it
Send a leaf photo to your bot. Example reply:
```
🌱 Crop Disease Diagnosis

🔍 Disease: Tomato Late blight
🌿 रोग: टमाटर - पछेती झुलसा रोग
📊 Confidence: 97.8%

💊 Please check the treatment guide for this condition.
```

---

## 📊 Model Performance

- **Validation Accuracy:** 97.75%
- Improved real-world generalization via data augmentation (random crop, flip, rotation, color jitter) after initial testing revealed a lab-vs-field domain gap.

---

## 🔭 Future Scope

- Pest detection and disease severity estimation
- Additional regional languages
- Environmental data integration (soil, weather) for early-warning prediction
- Edge/offline deployment (TensorFlow Lite)
- Expanded crop and disease coverage

---

## 👥 Team

- Divya Prakash Tiwari (301411123033)
- Aditya Naidu (301411123003)
- Yasir Ashraf (301411123031)
- MD. Sakib (301411123014)

**Guided by:** Dr. Abha Choubey
**Department of Computer Science & Engineering (IoT), Shri Shankaracharya Technical Campus, Bhilai**
