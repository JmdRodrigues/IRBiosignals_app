import streamlit as st
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

from tools.ssm_tools import *

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    s = np.loadtxt(uploaded_file)

st.title("NOVA App - Time Series Segmentation and Annotation")

#load data
st.line_chart(s)

#put params
window_size = st.slider('window size', 10, len(s)//3, 500)

S = compute_ssm(s, window_size, 0.95)
# im = Image.fromarray(S).convert("RGB")
# figure = plt.figure()
im = plt.imshow(S, aspect="auto")
# st.pyplot(figure)
norm = matplotlib.colors.Normalize(vmin=np.min(S), vmax=np.max(S), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
node_color = np.array([[(r, g, b) for r, g, b, a in mapper.to_rgba(S[:, i])] for i in range(0, len(S))])
st.image(node_color, use_column_width="always")

kernel_size = st.slider('kernel size', 2, len(S)//3, int(250/(len(s)/len(S))))

nov_ssm = compute_novelty(S, kernel_size)

st.line_chart(nov_ssm)
