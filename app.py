import kraken
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib import models
from kraken import blla, rpred
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 
import torch

#title
st.title("BiblIA OCR")

#subtitle
st.markdown("## Hebrew handwritten text recognition using `kraken`, `streamlit`")

st.markdown("")

col1, col2 = st.columns(2)

col1.header("Image")
col2.header("Text")

# #image uploader
# image = col1.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])
image = None

@st.cache
def load_seg_model(): 
    model_path = 'models/seg/biblialong02_se3_2_tl.mlmodel'
    model = TorchVGSLModel.load_model(model_path)
    model.eval()
    return model 

@st.cache
def load_rec_model(): 
    rec_model_path = 'models/rec/biblia_tr_9.mlmodel'
    model = models.load_any(rec_model_path)
    # model.eval()
    return model

seg_model = load_seg_model()
rec_model = load_rec_model()

if image is None:
    with col1:     
        #image uploader
        image = col1.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])
    with col2:
        st.write("Upload an Image")

if image is not None:
    input_image = Image.open(image) #read image
    
    # st.image(input_image) #display image
    with col1:
        st.image(input_image, use_column_width=True)

    with col2:
        with st.spinner("🤖 AI is at Work! "):
            with torch.no_grad():
                baseline_seg = blla.segment(input_image, model=seg_model, text_direction='rl')
                pred_it = rpred.rpred(rec_model, input_image, baseline_seg)
                result_text = []
                for record in pred_it:
                    t = str(record)
                    result_text.append(t)

            result_text_joined = "\n".join(result_text)
            st.markdown("""
            <style>
            input {
            unicode-bidi:bidi-override;
            direction: RTL;
            }
            </style>
                """, unsafe_allow_html=True)
            st.write(result_text_joined)
    #st.success("Here you go!")
    st.balloons()
    
    

st.caption("Code by @anutkk, credit for baseline code to @1littlecoder's code.")




