from lavis.models import load_model_and_preprocess
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="caption_coco_opt2.7b",
    is_eval=True,
    device=device
)

raw_image = Image.open("data/val2017/000000039769.jpg").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

caption = model.generate({"image": image})
print("Caption:", caption[0])
