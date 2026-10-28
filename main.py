from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
import json

app = FastAPI()

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)
with open("class_names_hindi.json") as f:
    class_names_hindi = json.load(f)

print("Loaded classes:", len(class_names))


# Select GPU if available, otherwise use CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create the same model architecture
model = models.mobilenet_v2(weights=None)


# Change the final layer to match the number of classes
model.classifier[1] = torch.nn.Linear(
    model.last_channel,
    len(class_names)
)


# Load trained model weights
model.load_state_dict(
    torch.load(
        "crop_disease_model.pth",
        map_location=device
    )
)


# Move model to selected device
model.to(device)


# Set model to evaluation mode
model.eval()


print("Model loaded successfully")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    disease = class_names[predicted.item()]
    disease_hindi = class_names_hindi.get(disease, disease)
    conf_score = confidence.item()

    return {
        "disease": disease,
        "disease_hindi": disease_hindi,
        "confidence": round(conf_score, 4)
    }