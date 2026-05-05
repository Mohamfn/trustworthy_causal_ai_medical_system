import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import time

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Trustworthy Causal AI Medical System",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fb;
}
.big-title {
    text-align: center;
    color: #12355b;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #4a4a4a;
    font-size: 17px;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.warning-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #ffe6e6;
    border-left: 7px solid #cc0000;
    font-size: 18px;
}
.success-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #e6ffe6;
    border-left: 7px solid green;
    font-size: 18px;
}
.info-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #e8f1ff;
    border-left: 7px solid #1f77b4;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="big-title">🏥 Trustworthy Causal AI Medical System</div>
<div class="subtitle">
Brain Tumor Detection • Lung Cancer Detection • Dementia Prediction • Explainability • Causal AI
</div>
<hr>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏥 Hospital Navigation")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Brain Tumor Detection",
        "Lung Cancer Detection",
        "Dementia Prediction",
        "Causal AI",
        "About Project"
    ]
)

# ---------------- HELPER FUNCTIONS ----------------
def loading_effect(message):
    with st.spinner(message):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)


def display_heatmap(image):
    img = np.array(image.resize((224, 224)))

    heatmap = np.mean(img, axis=2)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img)
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.axis("off")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    st.caption("Simulated Grad-CAM style heatmap using intensity-based approximation.")


def hospital_warning(text, confidence):
    st.markdown(f"""
    <div class="warning-card">
    ⚠️ <b>{text}</b><br>
    AI Confidence Score: <b>{confidence}%</b>
    </div>
    """, unsafe_allow_html=True)


def hospital_success(text, confidence=None):
    if confidence:
        st.markdown(f"""
        <div class="success-card">
        ✅ <b>{text}</b><br>
        AI Confidence Score: <b>{confidence}%</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-card">
        ✅ <b>{text}</b>
        </div>
        """, unsafe_allow_html=True)


# ---------------- HOME ----------------
if menu == "Home":
    st.header("📋 Project Overview")

    st.markdown("""
    <div class="card">
    This project presents a <b>Trustworthy Causal AI healthcare prototype</b> that integrates 
medical image analysis, tabular risk prediction, explainability, and causal reasoning. 
The goal is not to replace clinicians, but to demonstrate how AI systems can support 
transparent and interpretable decision-making in healthcare.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🧠 Brain Tumor Detection</h3>
        Simulated CNN-based brain scan analysis with heatmap explanation.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🫁 Lung Cancer Detection</h3>
        Simulated CNN-based lung CT/X-ray diagnosis with visual explanation.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
        <h3>🧓 Dementia Prediction</h3>
        Random Forest model predicts dementia risk using patient data.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <b>Trustworthy AI Components:</b><br>
    🔍 Explainability using heatmaps and feature importance<br>
    🔄 Causal reasoning using what-if interventions<br>
    📊 Transparent prediction results<br>
    🏥 Hospital-style clinical decision support interface
    </div>
    """, unsafe_allow_html=True)


# ---------------- BRAIN TUMOR ----------------
elif menu == "Brain Tumor Detection":
    st.header("🧠 Brain Tumor Detection")

    st.markdown("""
    <div class="card">
    Upload a brain CT/MRI image. The system simulates CNN-based image diagnosis and provides an explainability heatmap.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Brain CT/MRI Image",
        type=["jpg", "jpeg", "png"],
        key="brain_upload"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Brain Scan", width=280)

        loading_effect("Analyzing brain scan using simulated CNN model...")

        st.subheader("📊 Prediction Result")
        hospital_warning("Possible Brain Tumor Detected", 87)

        st.subheader("🔥 Explainability Module")
        st.write("The scan is shown with an explanation note below. Real Grad-CAM can be added in a future version.")
        display_heatmap(image)

        st.info("""
        Academic Prototype Note:
        This module demonstrates a CNN-style medical image analysis workflow for brain tumor detection. 
        The prediction output is simulated for demonstration purposes, while the system focuses on 
        trustworthy AI principles such as transparency, visual explainability, and clinical decision support.

        In a production version, this module would be connected to a trained CNN model such as ResNet, VGG, 
        or EfficientNet and validated using real medical imaging datasets.
        """)


# ---------------- LUNG CANCER ----------------
elif menu == "Lung Cancer Detection":
    st.header("🫁 Lung Cancer Detection")

    st.markdown("""
    <div class="card">
    Upload a lung CT/X-ray image. The system simulates CNN-based lung abnormality detection and provides a heatmap explanation.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Lung CT/X-ray Image",
        type=["jpg", "jpeg", "png"],
        key="lung_upload"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Lung Scan", width=280)

        loading_effect("Analyzing lung scan using simulated CNN model...")

        st.subheader("📊 Prediction Result")
        hospital_warning("Possible Lung Abnormality / Cancer Risk Detected", 84)

        st.subheader("🔥 Explainability Heatmap/Region of Interest")
        st.write("The heatmap regions shows the area emphasized by the prototype model during interpretation.")
        display_heatmap(image)

        st.info("""
        Academic Prototype Note:
        This module demonstrates a CNN-style lung CT/X-ray analysis workflow. 
        The prediction result and confidence score are simulated to represent how a trained medical AI model 
        would communicate risk to clinicians.

        The heatmap provides a visual explanation layer, similar in purpose to Grad-CAM, showing which image 
        regions are emphasized during interpretation.
        """)


# ---------------- DEMENTIA ----------------
elif menu == "Dementia Prediction":
    st.header("🧓 Dementia Risk Prediction")

    st.markdown("""
    <div class="card">
    Enter patient information below. A Random Forest model estimates dementia risk using tabular clinical features.
    </div>
    """, unsafe_allow_html=True)

    age = st.slider("Age", 40, 100, 65, key="dementia_age")
    smoking = st.selectbox("Smoking History", ["No", "Yes"], key="dementia_smoking")
    bp = st.slider("Blood Pressure", 80, 200, 130, key="dementia_bp")
    diabetes = st.selectbox("Diabetes", ["No", "Yes"], key="dementia_diabetes")
    memory = st.slider("Memory Score (0–30)", 0, 30, 20, key="dementia_memory")

    smoking_value = 1 if smoking == "Yes" else 0
    diabetes_value = 1 if diabetes == "Yes" else 0

    np.random.seed(42)

    data = pd.DataFrame({
        "age": np.random.randint(40, 100, 300),
        "smoking": np.random.randint(0, 2, 300),
        "bp": np.random.randint(80, 200, 300),
        "diabetes": np.random.randint(0, 2, 300),
        "memory": np.random.randint(0, 30, 300)
    })

    data["risk"] = (
        (data["age"] > 70).astype(int) +
        (data["bp"] > 140).astype(int) +
        (data["diabetes"] == 1).astype(int) +
        (data["memory"] < 15).astype(int) +
        (data["smoking"] == 1).astype(int)
    )

    data["risk"] = (data["risk"] >= 2).astype(int)

    X = data.drop("risk", axis=1)
    y = data["risk"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    user = pd.DataFrame(
        [[age, smoking_value, bp, diabetes_value, memory]],
        columns=X.columns
    )

    if st.button("Predict Dementia Risk", key="predict_dementia_button"):
        loading_effect("Analyzing patient data using Random Forest model...")

        pred = model.predict(user)[0]
        probability = model.predict_proba(user)[0][1]

        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.markdown(f"""
            <div class="warning-card">
            🔴 <b>High Dementia Risk</b><br>
            Risk Probability: <b>{probability:.2f}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-card">
            🟢 <b>Low Dementia Risk</b><br>
            Risk Probability: <b>{probability:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("🔍 Feature Importance Explanation")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance.set_index("Feature"))

        st.info(
            "Memory Score represents a simplified cognitive score similar to MMSE, where lower scores indicate higher dementia risk."
        )


# ---------------- CAUSAL AI ----------------
elif menu == "Causal AI":
    st.header("🔄 Causal What-If Analysis")

    st.markdown("""
    <div class="card">
    This module demonstrates causal reasoning. Instead of only predicting risk, it asks:
    <b>What would happen if a patient changed one or more risk factors?</b>
    </div>
    """, unsafe_allow_html=True)

    age = st.slider("Age", 40, 100, 75, key="causal_age")
    bp = st.slider("Blood Pressure", 80, 200, 150, key="causal_bp")
    smoking = st.selectbox("Smoking", ["Yes", "No"], key="causal_smoking")
    exercise = st.selectbox("Regular Exercise", ["No", "Yes"], key="causal_exercise")

    original_risk = 0

    if age > 70:
        original_risk += 1
    if bp > 140:
        original_risk += 1
    if smoking == "Yes":
        original_risk += 1
    if exercise == "No":
        original_risk += 1

    st.subheader("📌 Original Risk")
    st.write(f"Original Risk Score: **{original_risk}/4**")

    st.subheader("🔁 Counterfactual Intervention")
    st.write("Intervention: Patient stops smoking and starts exercising.")

    new_risk = 0

    if age > 70:
        new_risk += 1
    if bp > 140:
        new_risk += 1

    st.write(f"New Risk Score After Intervention: **{new_risk}/4**")

    if new_risk < original_risk:
        st.success("Risk reduced after intervention.")
    else:
        st.warning("Risk did not reduce significantly.")

    st.subheader("🧩 Causal Graph")

    G = nx.DiGraph()
    G.add_edges_from([
        ("Age", "Health Risk"),
        ("Blood Pressure", "Health Risk"),
        ("Smoking", "Health Risk"),
        ("Exercise", "Health Risk"),
        ("Health Risk", "Clinical Outcome")
    ])

    fig, ax = plt.subplots(figsize=(6, 4))

    pos = nx.spring_layout(G, seed=42, k=1.2)

    labels = {
        "Age": "Age",
        "Blood Pressure": "Blood\nPressure",
        "Smoking": "Smoking",
        "Exercise": "Exercise",
        "Health Risk": "Health\nRisk",
        "Clinical Outcome": "Clinical\nOutcome"
    }

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=4200,
        node_color="#9fd3ff",
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=18,
        width=2,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
        ax=ax
    )

    ax.margins(0.25)
    ax.axis("off")
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.info("""
    Causal Interpretation:
    This module demonstrates intervention-based reasoning. Instead of only predicting risk from existing data,
    the system asks a counterfactual question: what would happen if modifiable risk factors changed?

    This supports Trustworthy Causal AI by showing:
    - cause-effect relationships,
    - transparent reasoning,
    - actionable clinical insight,
    - and the difference between correlation-based prediction and intervention-based analysis.
    """)


# ---------------- ABOUT ----------------
elif menu == "About Project":
    st.header("ℹ️ About This Project")

st.write("""
This project is based on the proposal: 
**Using CNNs and Random Forests for Diagnosing CT Scans and Dementia Prediction with Trustworthy AI.**

The system demonstrates a multi-module healthcare AI prototype with the following components:

1. **Brain Tumor Detection**  
   A CNN-style image analysis workflow for brain scan interpretation.

2. **Lung Cancer Detection**  
   A CNN-style image analysis workflow for lung CT/X-ray interpretation.

3. **Dementia Risk Prediction**  
   A Random Forest classifier using patient-level tabular features such as age, blood pressure,
   diabetes status, smoking history, and memory score.

4. **Explainability**  
   Image heatmaps and Random Forest feature importance are included to make predictions more transparent.

5. **Causal AI**  
   The causal module demonstrates intervention-based reasoning by asking what happens when modifiable
   risk factors such as smoking and exercise are changed.

This is an academic prototype designed to demonstrate Trustworthy AI principles: transparency,
interpretability, causal reasoning, and responsible clinical decision support. It is not a real
medical diagnostic device.
""")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Developed by Mohammed Furquan Ghori | MS Computer Science | Trustworthy Causal AI Healthcare Prototype</p>",
    unsafe_allow_html=True
)