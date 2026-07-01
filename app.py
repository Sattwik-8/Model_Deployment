import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("iris_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

SPECIES = ["Setosa", "Versicolor", "Virginica"]
SPECIES_IMAGES = {
    "Setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    "Versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
}

# ---------------- Header ----------------
st.title("🌸 Iris Species Predictor")
st.markdown(
    "An end-to-end **Machine Learning + Deployment** project. "
    "Adjust the flower measurements in the sidebar and the model "
    "will predict the most likely Iris species in real time."
)

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        - **Model**: Logistic Regression classifier trained on the classic Iris dataset
        - **Input features**: Sepal length, Sepal width, Petal length, Petal width (cm)
        - **Output**: Predicted species + confidence across all 3 classes
        - **Stack**: scikit-learn (training) → Streamlit (UI) → Streamlit Cloud (deployment)
        """
    )

st.divider()

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔧 Input Flower Measurements")

sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Species", use_container_width=True, type="primary")

# Live snapshot of current inputs
st.subheader("Current Input")
input_df = pd.DataFrame(
    {
        "Feature": ["Sepal length", "Sepal width", "Petal length", "Petal width"],
        "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width],
    }
)
st.dataframe(input_df, hide_index=True, use_container_width=True)

# ---------------- Prediction ----------------
if predict_btn:
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_species = SPECIES[prediction] if isinstance(prediction, (int, np.integer)) else prediction

    st.divider()
    st.subheader("🎯 Prediction Result")

    col1, col2 = st.columns([1, 2])
    with col1:
        img_url = SPECIES_IMAGES.get(str(predicted_species).capitalize())
        if img_url:
            st.image(img_url, caption=predicted_species, use_container_width=True)
    with col2:
        st.success(f"**Predicted Species: {predicted_species}**")
        confidence = max(probabilities) * 100
        st.metric("Confidence", f"{confidence:.1f}%")

    st.markdown("#### Prediction Probabilities")
    prob_df = pd.DataFrame({"Species": SPECIES, "Probability": probabilities}).set_index("Species")
    st.bar_chart(prob_df)

else:
    st.info("👈 Set the measurements in the sidebar and click **Predict Species** to see results.")

# ---------------- Footer ----------------
st.divider()
st.caption("Built with Streamlit · scikit-learn · Deployed on Streamlit Community Cloud")