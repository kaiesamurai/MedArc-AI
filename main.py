import streamlit as st
import tempfile
from src.models.predict_label import get_prediction
from src.models.heatmap import *
import replicate
import tensorflow as tf

st.set_page_config(page_title='MedArc AI (using Snowflake Arctic)')
st.title("MedArc AI (using Snowflake Arctic)")

# Initialize prompt variable
prompt = ""

# Chose Replicate model
replicate_model = st.sidebar.selectbox('Model', ['snowflake-arctic'])

# Add uploader function
st.sidebar.subheader("File Uploader:", divider='rainbow')

uploaded_file = st.sidebar.file_uploader("Choose files",
                                         type=['.jpg', '.jpeg', '.png'],
                                         accept_multiple_files=False)
prediction = None
if uploaded_file:
    st.sidebar.image(uploaded_file, caption='Image uploaded by patient', width=100)
    # Perform classification using CNN model
    # Replace this with your CNN model's prediction code
    classification = None  # Replace with your classification code
    if classification:
        prediction = f"it's {classification[0][0]} with {classification[0][1]} probability"
        with st.container(border=True):
            st.sidebar.subheader('Probabilities of Classification', divider='rainbow')
            st.sidebar.metric(f':green[{classification[0][0]}]', value='{:.2f}%'.format(classification[0][1]))
            st.sidebar.metric(f':green[{classification[1][0]}]', '{:.2f}%'.format(classification[1][1]))
            st.sidebar.metric(f':green[{classification[2][0]}]', '{:.2f}%'.format(classification[2][1]))
        # Generate Grad-CAM heatmap
        heatmap = None  # Replace with your Grad-CAM heatmap generation code
        if heatmap:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
                temp_file_path = temp_file.name
                # Save and display Grad-CAM heatmap
                save_and_display_gradcam(uploaded_file, heatmap, cam_path=temp_file_path, alpha=0.8)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.write("Hello 👋, I am MedArc (Snowflake Arctic), I'm here to help you as an AI medical assistant")

# Input patient information: age, sex, history, and lesion characters
age = st.number_input('Age', step=1)
sex = st.multiselect('Sex', ['Male', 'Female', 'Other'], max_selections=1)
patient_history = st.text_input('Patient history')
lesion_char = st.text_input('Lesion characters')

# Prompt generation
intro = 'Hi, I need some information about the following information'
action = "Please give a definition of the disease, a short description of it, the level of danger and what are the actions to take. Answer in 4 different bullet points with the 4 different titles [Definition, Description, Threat, Actions to take]."
if prediction is not None:
    prompt = f'<s> [INST] {intro}. I am {age} years old, my sex is defined as {sex}. My medical history is {patient_history}. I have a {lesion_char}. According to CNN model prediction: {prediction}. Please provide detailed information about the disease, including its causes, symptoms, diagnosis, treatment options, and any preventive measures. Additionally, explain how this disease can be prevented. "[/INST]"'

# Generate response
if st.button('Generate', type="primary"):
    try:
        # Formatting the prompt
        prompt = "<s> [INST] " + prompt + " [/INST]"
        # Generate response using Replicate API
        for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                                      input={"prompt": prompt}):
            st.write(event)

    except Exception as err:
        st.exception('An error occurred while generating response!')
        print(err)
