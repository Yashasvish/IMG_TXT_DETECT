import PIL 
from PIL import ImageDraw
from PIL import Image
import streamlit as st
import os


def load_image(image_file):
	img = PIL.Image.open(image_file)
	return img

def init_session_states():
  if 'disp' not in st.session_state:
    st.session_state['disp'] = st.empty()
    st.session_state['disp'].text("Setting up environment with latest build of easyocr. This will take about a minute ")
  if 'init' not in st.session_state:
    st.session_state['init'] = 1





init_session_states()
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def text_recognition(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    pixel_values = processor(image, return_tensors="pt").pixel_values 
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
    st.write(generated_text)

def main():
   
    st.session_state['disp'].text("Env setup up Complete")
    uploaded_file = st.file_uploader("Choose image file to detect text",type=['jpeg','jpg'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
        image = load_image(uploaded_file)
        st.image(image,width=500)
        st.write("Detecting text bounding box and Take 1 recognition...")
        reader = easyocr.Reader(['en'],gpu=True)
        bound = reader.readtext(image)
        st.write("Bounding box Detection complete")
        st.write(str(bound))
        st.write("Recognizing text - Take 2....")
        text_recognition(image)
   
   
 
if __name__ == "__main__":
    main()
                  

   
    
