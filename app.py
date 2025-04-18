import streamlit as st
from src.utils import load_image
from src.backend import load_model, classify

st.set_page_config(page_title='Facial Gender Classifier')
st.header('Facial Gender Classifier')

model, transform = load_model(checkpoint_path='models/gender_classifier_250418.pt')

with st.form('main_form'):
    file = st.file_uploader(label='Choose an image here', type=['jpg', 'jpeg', 'png'])
    uploaded = st.form_submit_button()

if uploaded:
    with st.container(border=True):
        image = load_image(file)
        pred, prob = classify(image, model, transform, return_proba=True)
        st.image(image)
        st.markdown(f'**Prediction**: {pred}\n\n**Confidence**: {round(prob, 4)}')
        clear_but = st.button(label='Reset')
        if clear_but:
            st.empty()
