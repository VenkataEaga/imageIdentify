import streamlit as st
from PIL import Image
from model_vgg16 import predict

st.title("Identify object with AI model")
uploaded_file = st.file_uploader("Upload any JPG file...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Identifying the uploaded image ")
    label = predict(uploaded_file)
    #st.write("This Object in the image is ","%s (%.2f%%)" % (label[1], label[2]*100))
    st.write("This Object in the image is --->  \" ","%s" % (label[1])," \"")