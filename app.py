import kraken
from kraken.lib.vgsl import TorchVGSLModel
from kraken.lib import models
from kraken import blla, rpred
import streamlit as st  #Web App
from PIL import Image, ImageDraw #Image Processing
import numpy as np #Image Processing 
import torch
from itertools import cycle
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# st.set_page_config(layout="wide")

#title
st.title("BiblIA OCR")

#subtitle
st.markdown("## Hebrew handwritten text recognition")

st.markdown("")

# #image uploader
holder = st.empty()
image = holder.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

@st.cache
def load_seg_model(): 
    model_path = 'models/seg/biblialong02_se3_2_tl.mlmodel'
    model = TorchVGSLModel.load_model(model_path)
    model.eval()
    return model 

@st.cache(allow_output_mutation=True)
def load_rec_model(): 
    rec_model_path = 'models/rec/biblia_tr_9.mlmodel'
    model = models.load_any(rec_model_path)
    # model.eval()
    return model

seg_model = load_seg_model()
rec_model = load_rec_model()

#allow for selecting an area
if image is not None:

    st.markdown(
        """
    * The image will be entirely processed below, in the Segmentation and Transcription sections.
    * If you wish to crop the image and rerun, choose an area in the following canvas and click the "Send to Streamlit" (lefmost) icon.
    """
    )

    input_image = Image.open(image) #read image
    display_width = 500
    display_height = input_image.size[1]/input_image.size[0]*display_width
    col1, col2 = st.columns([0.1, 1])
    with col2:
        canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                background_image=input_image,
                stroke_width=5,
                update_streamlit=False,
                width=display_width,
                height=display_height,
                drawing_mode="rect",
                display_toolbar=True,
                key="canvas",
            )
    holder.empty()

    # extract coordinates of rectangle and crop
    if (canvas_result.json_data is not None):
        # st.write(canvas_result.json_data)
        if len(canvas_result.json_data["objects"])>0:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            rect = objects.iloc[0]
            left, upper, width, height = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])

            #scale according to displayed size
            width_f = input_image.size[1] / display_height
            height_f = input_image.size[0] / display_width

            (left, upper, right, lower) = (width_f*left, height_f*upper, width_f*(left+width), height_f*(upper+height))
            
            im_crop = input_image.crop((left, upper, right, lower))
        else:
            im_crop = input_image
        
        #Process segmentation and transcription    
        col1, col2 = st.columns(2)

        col1.header("Segmentation")
        col2.header("Transcription")
        with col1:
            with st.spinner("🤖 AI is at Work! "):
                #compute and display segmentation
                baseline_seg = blla.segment(im_crop, model=seg_model, text_direction='rl')
                bmap = (0, 130, 200, 255)
                cmap = cycle([(230, 25, 75, 127),
                            (60, 180, 75, 127)])
                im = im_crop.convert('RGBA')
                tmp = Image.new('RGBA', im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(tmp)
                for  line in baseline_seg['lines']:
                        c = next(cmap)
                        if line['boundary']:
                            draw.polygon([tuple(x) for x in line['boundary']], fill=c, outline=c[:3])
                        # if line['baseline']:
                        #     draw.line([tuple(x) for x in line['baseline']], fill=bmap, width=2, joint='curve')
                        # draw.text(line['baseline'][0], str(idx), fill=(0, 0, 0, 255))
                base_image = Image.alpha_composite(im, tmp)
                st.image(base_image, use_column_width=True)

        with col2:
            with st.spinner("🤖 AI is at Work! "):
                with torch.no_grad():
                    # baseline_seg = blla.segment(input_image, model=seg_model, text_direction='rl')
                    pred_it = rpred.rpred(rec_model, input_image, baseline_seg)
                    result_text = []
                    for record in pred_it:
                        t = str(record)
                        result_text.append(t)

                result_text_joined = "<br>".join(result_text)
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

    
st.caption("Backend: [BiblIA model](https://zenodo.org/record/5468286#.Y0hD2XZByUk) using [kraken OCR system](https://kraken.re/master/index.html)")
st.caption("Academic reference: Stökl Ben Ezra, D., Brown-DeVost, B., Jablonski, P., Kiessling, B., Lolli, E., Lapin, H. “BiblIA – a General Model for Medieval Hebrew Manuscripts and an Open Annotated Dataset” HIP@ICDAR 2021")


st.caption("Code by [Shmuel Londner](https://github.com/anutkk).")




