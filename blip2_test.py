from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

print("Loading processor and model...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
print("Processor and model loaded!")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

print("Opening image...")
image = Image.open("test_image.jpeg").convert("RGB")

print("Processing inputs...")
inputs = processor(images=image, text="What is happening in this photo?", return_tensors="pt").to(device)

print("Generating output...")
outputs = model.generate(**inputs)

print("Decoding output...")
caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Caption:", caption)
