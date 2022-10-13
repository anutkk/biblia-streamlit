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

# #image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

col1, col2 = st.columns(2)

col1.header("Image")
col2.header("Text")

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
            .rtl {
            unicode-bidi:bidi-override;
            direction: RTL;
            }
            </style>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="rtl">%s</div>' % result_text_joined, unsafe_allow_html=True)
            # st.write(result_text_joined)
    #st.success("Here you go!")
    st.balloons()
else:
    with col2:
        st.write("Upload an Image")
    
st.caption("Backend: [BiblIA model](https://zenodo.org/record/5468286#.Y0hD2XZByUk) using [kraken OCR system](https://kraken.re/master/index.html)")
st.caption("Stökl Ben Ezra, Daniel; Brown-DeVost, Bronson; Jablonski, Pawel; Kiessling, Benjamin; Lolli, Elena; Lapin, Hayim: \"BiblIA - a general model for Medieval Hebrew manuscripts and an open annotated dataset\" published in The 6th International Workshop on Historical Document Imaging and Processing (HIP '21) (HIP@ICDAR), Lausanne, Switzerland, September 5-6, 2021")


st.caption("Code by [Shmuel Londner](https://github.com/anutkk).")




