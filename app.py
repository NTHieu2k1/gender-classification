import streamlit as st
from src.utils import load_image
from src.backend import upload_n_classify

st.set_page_config(page_title='Facial Gender Classifier', layout='wide')
st.header('Facial Gender Classifier')

# model, transform = load_model(checkpoint_path='models/gender_classifier_250418.pt')

col1, col2 = st.columns([1, 1])
with col1:
    form = st.form('main_form')
with col2:
    image_area = st.container(border=True, height=400)
pred_area = st.container(border=True, height=100)

with form:
    file = st.file_uploader(label='Choose an image here', type=['jpg', 'jpeg'])
    uploaded = st.form_submit_button()

if uploaded:
    with st.container(border=True):
        image, request_data = load_image(file)
        image_area.image(image)
        response = upload_n_classify(request_data)
        pred, prob = response['pred'], response['confidence']
        pred_area.markdown(f'**Prediction**: {pred}\n\n**Confidence**: {round(prob, 4)}')
        clear_but = pred_area.button(label='Reset')
        if clear_but:
            st.empty()
