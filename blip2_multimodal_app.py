import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ----------- Streamlit page config must be first -----------
st.set_page_config(page_title="BLIP-2 Multimodal App", layout="wide")

# ----------- Load model and processor -----------
@st.cache_resource(show_spinner=True)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    model.eval()
    return processor, model, device

processor, model, device = load_model()

# ----------- Helper functions -----------

def safe_cast_inputs(inputs_dict):
    """Only casts float tensors to float16/float32 based on device."""
    safe_inputs = {}
    for k, v in inputs_dict.items():
        v = v.to(device)
        if v.dtype.is_floating_point and device == "cuda":
            v = v.to(dtype=torch.float16)
        safe_inputs[k] = v
    return safe_inputs

def generate_caption(image):
    inputs = processor(images=image, text="", return_tensors="pt")
    inputs = safe_cast_inputs(inputs)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

def answer_question(image, question):
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = safe_cast_inputs(inputs)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

def image_text_similarity(image, text):
    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = safe_cast_inputs(image_inputs)

    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)

    text_inputs = processor(text=text, return_tensors="pt", padding=True)
    text_inputs = safe_cast_inputs(text_inputs)

    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)

    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    similarity = (image_embeds * text_embeds).sum(dim=-1).item()
    return similarity

# ----------- Session state for chat -----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------- Streamlit UI -----------
st.title("🧠 BLIP-2 Multimodal AI Vision Demo")

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    task = st.selectbox(
        "🔍 Select Task",
        [
            "Image Captioning",
            "Visual Question Answering (VQA)",
            "Image-Text Retrieval (Similarity Score)",
            "Visual Commonsense Reasoning (Why questions)",
            "Multimodal Dialog (Chat with image)",
        ],
    )

    if task == "Image Captioning":
        st.subheader("🖼️ Caption Generation")
        if st.button("Generate Caption"):
            caption = generate_caption(image)
            st.markdown(f"**Caption:** {caption}")

    elif task == "Visual Question Answering (VQA)":
        st.subheader("❓ Ask a question about the image")
        question = st.text_input("Enter your question", "")
        if question and st.button("Get Answer"):
            answer = answer_question(image, question)
            st.markdown(f"**Answer:** {answer}")

    elif task == "Image-Text Retrieval (Similarity Score)":
        st.subheader("🔗 Check similarity between image and text")
        text = st.text_input("Enter text to compare with image", "")
        if text and st.button("Compute Similarity"):
            similarity = image_text_similarity(image, text)
            st.markdown(f"**Similarity score:** {similarity:.4f} (cosine similarity)")

    elif task == "Visual Commonsense Reasoning (Why questions)":
        st.subheader("🧠 Ask a WHY question about the image")
        why_question = st.text_input("Enter your WHY question", "Why is this happening?")
        if why_question and st.button("Get Answer"):
            answer = answer_question(image, why_question)
            st.markdown(f"**Answer:** {answer}")

    elif task == "Multimodal Dialog (Chat with image)":
        st.subheader("💬 Chat with the image")
        user_input = st.text_input("Type your message (question/comment)", key="chat_input")

        if user_input and st.button("Send", key="send_button"):
            dialog_text = ""
            for entry in st.session_state.chat_history:
                dialog_text += f"User: {entry['user']}\nAI: {entry['ai']}\n"
            dialog_text += f"User: {user_input}\nAI:"

            inputs = processor(images=image, text=dialog_text, return_tensors="pt")
            inputs = safe_cast_inputs(inputs)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            response = processor.decode(outputs[0], skip_special_tokens=True)

            st.session_state.chat_history.append({"user": user_input, "ai": response})
            st.markdown(f"**AI:** {response}")

        if st.session_state.chat_history:
            st.markdown("### 🗂️ Chat History")
            for entry in st.session_state.chat_history:
                st.markdown(f"**User:** {entry['user']}")
                st.markdown(f"**AI:** {entry['ai']}")

else:
    st.info("👆 Please upload an image to begin.")
