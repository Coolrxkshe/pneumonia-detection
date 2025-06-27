import torch
import torchvision.transforms as transforms
from PIL import Image
from cnn_model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("saved_model.pth", map_location=device))
model.eval()

# Prediction function
def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return "PNEUMONIA" if predicted.item() == 1 else "NORMAL"

# Test
print(predict_image("chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"))
