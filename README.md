# Iris-ML-Deployment

An **Iris Species Classifier** built with **scikit-learn**, deployed as an interactive web app using **Streamlit**, showing the full pipeline from a trained model to a live, usable tool.

**Live App:** [1stmodel.streamlit.app](https://1stmodel.streamlit.app/)

---

## Features
- **Trained ML Model** – Logistic Regression classifier trained on the classic Iris dataset
- **Interactive UI** – Real-time sliders for all 4 input features, no coding required
- **Multi-Class Prediction** – Classifies flowers into 3 species: Setosa, Versicolor, Virginica
- **Confidence Output** – Prediction probabilities displayed as a bar chart, not just a single label
- **Live Deployment** – Hosted and publicly accessible via Streamlit Community Cloud
- **Cached Model Loading** – `@st.cache_resource` used so the model loads once, not per interaction

---

## Components / Tools Used
| Tool / Library         | Purpose                              |
|-------------------------|---------------------------------------|
| Python                  | Core language                        |
| scikit-learn             | Model training (Logistic Regression) |
| Streamlit                | Interactive web UI                   |
| Pandas / NumPy           | Data handling & feature formatting   |
| Streamlit Community Cloud| Deployment platform                  |

---

## How It Works
1. **Training** – A Logistic Regression model is trained on the Iris dataset (sepal/petal length & width → species) and serialized with `pickle` as `iris_model.pkl`.
2. **Model Loading** – The Streamlit app loads the pickled model once at startup and caches it in memory.
3. **User Input** – Four sliders in the sidebar capture sepal length, sepal width, petal length, and petal width from the user.
4. **Feature Packaging** – Inputs are assembled into a NumPy array matching the model's expected input shape.
5. **Prediction** – On button click, `model.predict()` returns the predicted species and `model.predict_proba()` returns confidence across all 3 classes.
6. **Result Display** – The predicted species and a probability bar chart are rendered back to the user instantly.

---

## Input Features
| Feature       | Range Used in UI | Unit |
|----------------|------------------|------|
| Sepal length   | 4.0 – 8.0         | cm   |
| Sepal width    | 2.0 – 4.5         | cm   |
| Petal length   | 1.0 – 7.0         | cm   |
| Petal width    | 0.1 – 2.5         | cm   |

## Output Classes
| Class | Species     |
|-------|-------------|
| 0     | Setosa      |
| 1     | Versicolor  |
| 2     | Virginica   |

---

## Sample Screenshot
Below: the app UI showing an input configuration and the resulting prediction with confidence breakdown.

![App Screenshot](screenshots/app_ui.png)

---

## How to Run

**Option A – Use the Live App**
Visit [1stmodel.streamlit.app](https://1stmodel.streamlit.app/) — no setup required.

**Option B – Run Locally**
```bash
git clone https://github.com/Sattwik-8/Model_Deployment.git
cd Model_Deployment
pip install -r requirements.txt
streamlit run app.py
```

---

## Notes
This project separates **model training** (offline, producing `iris_model.pkl`) from **model serving** (the Streamlit app), reflecting the real-world distinction between **model development** and **model deployment** as separate stages of an ML pipeline.
