import streamlit as st
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ✅ 1. Set page config (must be the first Streamlit command)
st.set_page_config(page_title="BLIP-2 AI Vision", layout="centered")

# ✅ 2. Load BLIP-2 model and processor
@st.cache_resource
def load_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model, device

processor, model, device = load_model()

# ✅ 3. Title and file uploader
st.title("🧠 BLIP-2: Vision + Language AI")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ✅ 4. Handle image upload
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ✅ 5. Automatic Caption Generation
    with st.spinner("📝 Generating Caption..."):
        inputs = processor(image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
        caption_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0].strip()
        st.info(f"📝 Caption: {caption}")

    # ✅ 6. Visual Question Answering (VQA)
    question = st.text_input("❓ Ask a question about this image:")

    if question:
        with st.spinner("🤖 Thinking..."):
            vqa_inputs = processor(image, question, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
            generated_ids = model.generate(**vqa_inputs, max_new_tokens=50)
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            st.success(f"🗣️ Answer: {answer}")
