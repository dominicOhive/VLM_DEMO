import streamlit as st
import base64
import io
from PIL import Image
# Import your actual core functions from the code you were given
from grounding import run_grounding_detection 
from queue_worker import _enrich_spatial_outputs

st.title("🔍 Spatial Vision AI Inspector")
st.write("Upload an image and type what you want the LVM to find.")

# 1. UI Elements
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
user_prompt = st.text_input("What should the AI look for?", "Find any cracks or dents")

if uploaded_file is not None and st.button("Analyze Image"):
    # Load and prepare image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convert image to Base64 string (which your grounding.py code expects)
    buffered = io.BytesIO()
    uploaded_file.seek(0)
    buffered.write(uploaded_file.read())
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    st.info("🧠 Brainstorming vision plan and running segmentations...")
    
    # Run this asynchronously using st.spinner
    import asyncio
    async def run_ai():
        return await run_grounding_detection(
            image_base64=img_base64,
            image_size=image.size,
            condition_name=user_prompt,
        )
    
    parsed, raw_text = asyncio.run(run_ai())
    
    # Run your clustering and masking functions
    detections = parsed.get("detections", [])
    _enrich_spatial_outputs(img_base64, detections, image.size)
    
    # Display results
    st.success("Analysis Complete!")
    st.subheader("AI Findings Summary")
    st.write(parsed.get("summary", "No summary returned."))
    
    st.subheader("Raw Output Detections JSON")
    st.json(detections)