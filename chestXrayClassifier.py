import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms


m = getattr(models, "resnet18")(weights="DEFAULT")  

m.fc = nn.Sequential(
    nn.Linear(m.fc.in_features, 1),
    nn.Sigmoid()
)

device = "cuda" if torch.cuda.is_available() else "cpu"
m.to(device)

m.load_state_dict(torch.load("dryWetClassification/model_weights.pth", map_location="cpu"))
m.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),                       # fixed input size
    transforms.ToTensor(),                               # image -> tensor (grid of numbers) in [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet mean...
                         std=[0.229, 0.224, 0.225])    # ...and std
])

img = Image.open(r"C:\cac2026\dryWetClassification\test\3b81bcea-ee88-4c13-b4f6-2a53161d2b22.jpg").convert("RGB")
img = transform(img)
img = torch.stack([img]).to(device)


with torch.no_grad():  
    outputs = m(img)  
    outputs = outputs.squeeze(1).cpu().numpy()
    print(outputs)
    pneumoniaScore = outputs