import timeit
import torch, time
import clip
from PIL import Image
import lightning as L

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# start_time = time.time()
start_time = timeit.default_timer()

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

end_time = timeit.default_timer()

# Calculate the prediction time
prediction_time = end_time - start_time

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# print(f"Prediction time: {end_time - start_time} seconds")
print(f"Prediction time: {prediction_time:.3f} seconds")

# # Save the speed to "cpu_speed.txt" with 3 decimal points
# with open("cpu_speed.txt", "w") as file:
#     file.write(f"{prediction_time:.3f}")

# Save the speed to a file
device_type = "gpu" if device == "cuda" else "cpu"
with open(f"{device_type}_speed.txt", "w") as file:
    file.write(f"{prediction_time:.3f}")