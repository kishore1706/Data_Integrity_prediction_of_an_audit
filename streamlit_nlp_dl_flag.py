import streamlit as st
from nlp_dl_flag_production_code import DIFlagModel

MODEL_FILE = "nlp_DI_flag_model.sav"
VECTORIZER_FILE = "nlp_DI_flag_vectorizer.sav"

st.set_page_config(
    page_title="Audit data integrity prediction",
    page_icon="📋🔍",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("AUDIT DATA INTEGRITY PREDICTION")
st.write(
    "**Enter an observation description below and click Predict**"
)

with st.expander("How it works", expanded=False):
    st.write(
        "Considering the entered observation description,"
        "this app predicts whether the observation description is affected by **Data Integrity** or not."
    )

input_text = st.text_area(
    label="Observation description",
    value="",
    height=200,
    placeholder="Type or paste the full observation description here..."
)

predict_button = st.button("Predict")

model = DIFlagModel()
try:
    model.load_artifacts(
        model_path=MODEL_FILE,
        vectorizer_path=VECTORIZER_FILE,
    )
except Exception as err:
    st.error(f"Unable to load model artifacts: {err}")
    st.stop()

if predict_button:
    if not input_text.strip():
        st.warning("Please enter a description before predicting.")
    else:
        try:
            prediction = model.predict(input_text)
            st.success("Prediction complete")
            st.markdown("### Prediction result")
            st.write(f"**DL Flag Label:** {prediction['dl_flag_label']}")
            st.write(f"**Raw prediction:** {prediction['raw_prediction']}")
            if prediction['confidence'] is not None:
                st.write(f"**Confidence:** {prediction['confidence']:.2f}")
            st.write("If **Data Integrity Flag = 1**, There is a potential Data Integrity concern. The data may not be fully reliable.This does not always mean fraud or intentional wrongdoing. It simply means something happened that could affect confidence in the data.")
            st.write("If **Data Integrity Flag= 0**, No significant Data Integrity concern was identified.The observation may still be an issue, but it does not directly affect the trustworthiness of the data.")
            st.write("---")
            #st.markdown("### Processed text")
            #st.write(prediction['processed_text'])
        except Exception as err:
            st.error(f"Prediction failed: {err}")

st.sidebar.header("Instructions")
st.sidebar.write(
    "1. Paste the observation text in the main box.\n"
    "2. Click Predict,to see your **predicted result**\n"
)
