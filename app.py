import kraken
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib import models
from kraken import blla, rpred
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 

#title
st.title("BiblIA OCR")

#subtitle
st.markdown("## Hebrew handwritten text recognition using `kraken`, `streamlit`")

st.markdown("")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


@st.cache
def load_seg_model(): 
    model_path = 'models/seg/biblialong02_se3_2_tl.mlmodel'
    model = TorchVGSLModel.load_model(model_path)
    return model 

@st.cache
def load_rec_model(): 
    rec_model_path = 'models/rec/biblia_tr_9.mlmodel'
    model = models.load_any(rec_model_path)
    return model

seg_model = load_seg_model()
rec_model = load_rec_model()
if image is not None:
    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 AI is at Work! "):
        
        baseline_seg = blla.segment(input_image, model=seg_model, text_direction='rl')
        pred_it = rpred.rpred(rec_model, input_image, baseline_seg)
        result_text = []
        for record in pred_it:
            t = str(record)
            result_text.append(t)

        st.write(result_text)
    #st.success("Here you go!")
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Code by @anutkk, credit for baseline code to @1littlecoder's code.")




