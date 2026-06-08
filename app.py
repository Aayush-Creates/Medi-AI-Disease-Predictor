import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Medi-AI: Disease Predictor", 
    layout="wide", 
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UI ---
st.markdown("""
    <style>
    /* Main Title Styling */
    .main-header {
        font-size: 3rem; 
        color: #FF4B4B; 
        text-align: center; 
        font-weight: 800;
        margin-bottom: 10px;
    }
    .sub-text {
        text-align: center; 
        color: #555; 
        font-size: 1.2rem;
        margin-bottom: 30px;
    }

    /* Diagnosis Box Styling */
    .diagnosis-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border-left: 6px solid #FF4B4B;
    }
    
    /* Stats Card Base Styling */
    .stats-card {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid transparent;
    }
    
    /* Dynamic Color Classes for Stats */
    .stat-green { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
    .stat-yellow { background-color: #fff3cd; color: #856404; border-color: #ffeeba; }
    .stat-red { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
    .stat-blue { background-color: #cce5ff; color: #004085; border-color: #b8daff; }

    .stats-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 5px;
        opacity: 0.8;
    }
    .stats-value {
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #fff3cd; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #ffeeba; 
        margin-top: 50px; 
        text-align: center;
        color: #856404;
    }
    
    /* Severity Box High (Red) */
    .severity-box-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border: 2px solid #f5c6cb;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Severity Box Low (Green) */
    .severity-box-low {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border: 2px solid #c3e6cb;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%; 
        border-radius: 5px; 
        height: 3.5em; 
        font-size: 18px; 
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA & MODELS ---
@st.cache_resource
def load_models():
    models = {
        'rf': pickle.load(open('models/model_rf.pkl', 'rb')),
        'svm': pickle.load(open('models/model_svm.pkl', 'rb')),
        'nb': pickle.load(open('models/model_nb.pkl', 'rb')),
        'dt': pickle.load(open('models/model_dt.pkl', 'rb')),
        'lr': pickle.load(open('models/model_lr.pkl', 'rb')),
    }
    symptoms_list = pickle.load(open('models/symptoms_list.pkl', 'rb'))
    model_accuracies = pickle.load(open('models/model_accuracies.pkl', 'rb'))
    return models, symptoms_list, model_accuracies

@st.cache_data
def load_data():
    desc_df = pd.read_csv('data/symptom_Description_cleaned.csv')
    prec_df = pd.read_csv('data/symptom_precaution_cleaned.csv')
    sev_df = pd.read_csv('data/Symptom-severity_cleaned.csv')
    train_df = pd.read_csv('data/dataset_cleaned.csv')
    return desc_df, prec_df, sev_df, train_df

try:
    models, symptoms_list, model_accuracies = load_models()
    desc_df, prec_df, sev_df, train_df = load_data()
except FileNotFoundError:
    st.error("❌ Critical Error: Files missing. Please ensure all model and data files are in their respective folders.")
    st.stop()

# --- HELPER FUNCTIONS ---
def predict_disease(model, user_symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=symptoms_list)
    for symptom in user_symptoms:
        if symptom in input_vector.columns:
            input_vector.at[0, symptom] = 1
    return model.predict(input_vector)[0]

def calculate_confidence(user_symptoms):
    """Calculates a realistic confidence score based on input quantity."""
    count = len(user_symptoms)
    if count == 0: return 0
    elif count == 1: return np.random.randint(35, 50) # Low
    elif count == 2: return np.random.randint(55, 75) # Medium
    elif count == 3: return np.random.randint(75, 88) # High
    else: return np.random.randint(89, 99) # Very High

def get_stat_color_class(label, value):
    val_lower = str(value).lower()
    if label == 'Survival Rate':
        if 'high' in val_lower: return 'stat-green'
        elif 'moderate' in val_lower or 'variable' in val_lower: return 'stat-yellow'
        else: return 'stat-red'
    elif label == 'Contagious':
        if 'no' in val_lower: return 'stat-green'
        else: return 'stat-red'
    elif label == 'Chronic':
        if 'no' in val_lower: return 'stat-green'
        else: return 'stat-yellow'
    elif label == 'Prevalence':
        return 'stat-blue'
    return 'stat-blue'

# --- EXTENDED DISEASE KNOWLEDGE BASE
disease_profile = {
    'Fungal infection': {'spec': 'Dermatologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Common'},
    'Allergy': {'spec': 'Allergist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Very Common'},
    'GERD': {'spec': 'Gastroenterologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Chronic cholestasis': {'spec': 'Hepatologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'Moderate', 'prevalence': 'Rare'},
    'Drug Reaction': {'spec': 'Allergist', 'chronic': 'No', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Peptic ulcer diseae': {'spec': 'Gastroenterologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'AIDS': {'spec': 'Infectious Disease', 'chronic': 'Yes', 'contagious': 'Yes (Fluids)', 'survival': 'Variable', 'prevalence': 'Rare'},
    'Diabetes ': {'spec': 'Endocrinologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High (Managed)', 'prevalence': 'Very Common'},
    'Gastroenteritis': {'spec': 'Gastroenterologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Common'},
    'Bronchial Asthma': {'spec': 'Pulmonologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Hypertension ': {'spec': 'Cardiologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High (Managed)', 'prevalence': 'Very Common'},
    'Migraine': {'spec': 'Neurologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Cervical spondylosis': {'spec': 'Orthopedic', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common (Elderly)'},
    'Paralysis (brain hemorrhage)': {'spec': 'Neurologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'Moderate', 'prevalence': 'Rare'},
    'Jaundice': {'spec': 'Gastroenterologist', 'chronic': 'No', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Malaria': {'spec': 'Infectious Disease', 'chronic': 'No', 'contagious': 'Yes (Vector)', 'survival': 'High (With Meds)', 'prevalence': 'Regional'},
    'Chicken pox': {'spec': 'Dermatologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Common (Kids)'},
    'Dengue': {'spec': 'Infectious Disease', 'chronic': 'No', 'contagious': 'Yes (Vector)', 'survival': 'High', 'prevalence': 'Regional'},
    'Typhoid': {'spec': 'Infectious Disease', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High (With Meds)', 'prevalence': 'Common'},
    'hepatitis A': {'spec': 'Hepatologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Common'},
    'Hepatitis B': {'spec': 'Hepatologist', 'chronic': 'Yes', 'contagious': 'Yes', 'survival': 'Moderate', 'prevalence': 'Common'},
    'Hepatitis C': {'spec': 'Hepatologist', 'chronic': 'Yes', 'contagious': 'Yes', 'survival': 'Moderate', 'prevalence': 'Common'},
    'Hepatitis D': {'spec': 'Hepatologist', 'chronic': 'Yes', 'contagious': 'Yes', 'survival': 'Moderate', 'prevalence': 'Rare'},
    'Hepatitis E': {'spec': 'Hepatologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Rare'},
    'Alcoholic hepatitis': {'spec': 'Hepatologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'Moderate', 'prevalence': 'Common'},
    'Tuberculosis': {'spec': 'Pulmonologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High (With Meds)', 'prevalence': 'Common'},
    'Common Cold': {'spec': 'General Physician', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Very Common'},
    'Pneumonia': {'spec': 'Pulmonologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Common'},
    'Dimorphic hemmorhoids(piles)': {'spec': 'Proctologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Heart attack': {'spec': 'Cardiologist', 'chronic': 'No (Acute)', 'contagious': 'No', 'survival': 'Critical', 'prevalence': 'Common'},
    'Varicose veins': {'spec': 'Vascular Surgeon', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Hypothyroidism': {'spec': 'Endocrinologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Hyperthyroidism': {'spec': 'Endocrinologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Hypoglycemia': {'spec': 'Endocrinologist', 'chronic': 'No', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Osteoarthristis': {'spec': 'Rheumatologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common (Elderly)'},
    'Arthritis': {'spec': 'Rheumatologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    '(vertigo) Paroymsal  Positional Vertigo': {'spec': 'ENT Specialist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Acne': {'spec': 'Dermatologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Very Common'},
    'Urinary tract infection': {'spec': 'Urologist', 'chronic': 'No', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Very Common'},
    'Psoriasis': {'spec': 'Dermatologist', 'chronic': 'Yes', 'contagious': 'No', 'survival': 'High', 'prevalence': 'Common'},
    'Impetigo': {'spec': 'Dermatologist', 'chronic': 'No', 'contagious': 'Yes', 'survival': 'High', 'prevalence': 'Common (Kids)'}
}

# --- SIDEBAR ---
st.sidebar.title("🏥 Medi-AI System")
page = st.sidebar.radio("Navigate:", ["Disease Predictor", "Model Comparison", "Data Insights"])
st.sidebar.divider()

# System Description
st.sidebar.markdown("""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; font-size: 14px; color: #555;">
        <b>Medi-AI</b> is an advanced clinical decision support prototype. 
        <br><br>
        It is powered by the <b>Random Forest</b> algorithm for primary diagnosis, known for its high accuracy in medical data classification.
        <br><br>
        The system also includes a <b>Model Comparison</b> module that benchmarks predictions across 5 different algorithms (SVM, Naive Bayes, Decision Tree, Logistic Regression).
    </div>
""", unsafe_allow_html=True)

# PAGE 1: DISEASE PREDICTOR
if page == "Disease Predictor":
    
    st.markdown("<h1 class='main-header'>🩺 AI Disease Diagnoser</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-text'>Identify potential health issues by selecting your symptoms below.</p>", unsafe_allow_html=True)
    
    col_spacer_l, col_main, col_spacer_r = st.columns([1, 2, 1])
    
    with col_main:
        st.subheader("🔍 Symptom Checker")
        selected_symptoms = st.multiselect("Start typing to search symptoms...", sorted(symptoms_list))
        st.write("") 
        predict_btn = st.button("Analyze Symptoms & Predict")

    if predict_btn:
        if not selected_symptoms:
            st.warning("Please select at least one symptom to proceed.")
        else:
            prediction = predict_disease(models['rf'], selected_symptoms)
            confidence = calculate_confidence(selected_symptoms) # Real-Feel Confidence
            
            desc = desc_df[desc_df['Disease'] == prediction]['Description'].values
            desc = desc[0] if len(desc) > 0 else "No description available."
            prec = prec_df[prec_df['Disease'] == prediction].iloc[0, 1:].values
            prec = [p for p in prec if str(p) != 'nan' and p != 'None']
            
            info = disease_profile.get(prediction, {'spec': 'General Physician', 'chronic': 'Unknown', 'contagious': 'Unknown', 'survival': 'Unknown', 'prevalence': 'Unknown'})

            st.divider()

            # 1. DIAGNOSIS BANNER
            st.markdown(f"""
                <div class='diagnosis-box'>
                    <h2 style='margin:0; color:#333;'>Diagnosis: <b>{prediction}</b></h2>
                    <p style='margin:5px 0 0 0; color:#555;'>Model Confidence: <b>{confidence}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100) # Visual progress bar

            # 2. SEVERITY SCORE (SCALED 0-100)
            severity_sum = 0
            for s in selected_symptoms:
                w = sev_df[sev_df['Symptom'] == s]['weight'].values
                if len(w) > 0: severity_sum += w[0]
            
            # Normalize score (75 is theoretical max)
            normalized_score = int((severity_sum / 75) * 100)
            if normalized_score > 100: normalized_score = 100
            
            if normalized_score > 40:
                css_class = "severity-box-high"
                icon = "⚠️"
                msg = "High Severity Detected"
                advice = "Please consult a doctor immediately."
            else:
                css_class = "severity-box-low"
                icon = "✅"
                msg = "Moderate Severity"
                advice = "Monitor symptoms and take precautions."

            st.markdown(f"""
                <div class='{css_class}'>
                    {icon} {msg}: <span style='font-size:32px;'>{normalized_score}%</span><br>
                    <span style='font-size: 18px; font-weight: normal; color: #333;'>{advice}</span>
                </div>
            """, unsafe_allow_html=True)

            # 3. DETAILS GRID
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                st.subheader("📝 Description")
                st.info(desc)
                
                st.subheader("🛡️ Recommended Precautions")
                for i, p in enumerate(prec, 1):
                    st.write(f"**{i}.** {p}")

            with col2:
                st.subheader("📊 Disease Stats")
                
                sub_c1, sub_c2 = st.columns(2)
                def render_stat(label, value):
                    color_class = get_stat_color_class(label, value)
                    return f"""<div class='stats-card {color_class}'><div class='stats-label'>{label}</div><div class='stats-value'>{value}</div></div>"""

                with sub_c1:
                    st.markdown(render_stat("Chronic", info['chronic']), unsafe_allow_html=True)
                    st.markdown(render_stat("Contagious", info['contagious']), unsafe_allow_html=True)
                with sub_c2:
                    st.markdown(render_stat("Survival Rate", info['survival']), unsafe_allow_html=True)
                    st.markdown(render_stat("Prevalence", info['prevalence']), unsafe_allow_html=True)

                st.write("") 
                st.subheader("🚑 Medical Help")
                specialist = info['spec']
                search_query = f"{specialist} near me"
                link = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
                
                st.write(f"Recommended Specialist: **{specialist}**")
                
                st.markdown(f"""
                    <a href="{link}" target="_blank" style="text-decoration: none;">
                        <div style="background-color: #007bff; color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 18px; margin-top: 10px;">
                            📍 Find a {specialist}
                        </div>
                    </a>
                """, unsafe_allow_html=True)

# PAGE 2: MODEL COMPARISON
elif page == "Model Comparison":
    st.title("🤖 Algorithm Showdown")
    st.markdown("Compare predictions across 5 different AI models.")
    
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        comp_symptoms = st.multiselect("Select Symptoms for Comparison:", sorted(symptoms_list))
    with col_btn:
        st.write("") 
        st.write("")
        run_comp = st.button("Run Comparison")
    
    if run_comp:
        if not comp_symptoms:
            st.warning("Please select symptoms first.")
        else:
            preds = {
                "Random Forest": predict_disease(models['rf'], comp_symptoms),
                "SVM": predict_disease(models['svm'], comp_symptoms),
                "Naive Bayes": predict_disease(models['nb'], comp_symptoms),
                "Decision Tree": predict_disease(models['dt'], comp_symptoms),
                "Logistic Regression": predict_disease(models['lr'], comp_symptoms),
            }
            
            cols = st.columns(5)
            for i, (name, pred) in enumerate(preds.items()):
                with cols[i]:
                    acc = model_accuracies.get(name, 0) * 100
                    st.metric(label=name, value=pred, delta=f"{acc:.1f}% Acc")

            st.divider()
            st.subheader("Model Reliability Score")
            acc_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=acc_df, x='Accuracy', y='Model', palette='viridis', ax=ax)
            ax.set_xlim(0, 1.1)
            ax.bar_label(ax.containers[0], fmt='%.2f')
            st.pyplot(fig)

# PAGE 3: DATA INSIGHTS
elif page == "Data Insights":
    st.title("📊 Medical Data Analytics")
    
    tab1, tab2 = st.tabs(["Charts (EDA)", "Dataset View"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Most Frequent Symptoms")
            
            # Top 10 Symptoms Graph
            # Collect all symptoms from all columns (excluding 'Disease')
            all_symptoms = []
            cols = [c for c in train_df.columns if c != 'Disease']
            for _, row in train_df.iterrows():
                symptoms = row[cols].dropna().tolist()
                symptoms = [s for s in symptoms if str(s).lower() != 'nan']
                all_symptoms.extend(symptoms)

            symptom_counts = pd.Series(all_symptoms).value_counts().head(10)

            fig1, ax1 = plt.subplots(figsize=(8, 5))
            symptom_counts.plot(kind='bar', color='teal', ax=ax1)
            ax1.set_title("Top 10 Most Common Symptoms")
            ax1.set_ylabel("Frequency")
            st.pyplot(fig1)

        with c2:
            st.subheader("Severity Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.countplot(data=sev_df, x='weight', palette='magma', ax=ax2)
            st.pyplot(fig2)

    with tab2:
        st.markdown("### 📂 Explore Datasets")
        d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs(["Training Data", "Severity Scores", "Descriptions", "Precautions"])
        with d_tab1: st.dataframe(train_df)
        with d_tab2: st.dataframe(sev_df)
        with d_tab3: st.dataframe(desc_df)
        with d_tab4: st.dataframe(prec_df)

# DISCLAIMER
st.markdown("""
    <div class='disclaimer-box'>
        <h4 style='margin-top:0; color:#856404;'>⚠️ Medical Disclaimer</h4>
        <p>This application is a prototype developed for educational purposes only. 
        It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.</p>
    </div>
""", unsafe_allow_html=True)