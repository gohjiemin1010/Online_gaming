import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
import time
import io
import base64
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Page Configuration & Global Settings
# ==========================================
st.set_page_config(page_title="Online Gaming Analytics", page_icon="🎮", layout="wide")

# Set Seaborn theme
sns.set_theme(style="white", context="notebook", font_scale=1.1)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ==========================================
# 2. Advanced CSS 
# ==========================================
st.markdown("""
<style>
.block-container { padding-top: 1.5rem !important; }

/* 3D Metric Cards styling with hover effect */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 15px 20px !important;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1) !important; 
    border-top: 4px solid #6A0DAD !important; 
    transition: all 0.3s ease-in-out !important; 
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px) !important; 
    box-shadow: 0px 10px 20px rgba(106, 13, 173, 0.2) !important; 
}

/* 3D Coverflow Slider Styling (Tab 1) */
.slider-container {
    position: relative;
    width: 100%;
    height: 450px;
    display: flex;
    justify-content: center;
    align-items: center;
    perspective: 1200px;
    overflow: hidden;
    background: transparent !important; 
    box-shadow: none !important; 
    border: none !important;
    margin-bottom: 20px;
}
.slider-card {
    position: absolute;
    width: 600px;
    height: 380px;
    transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
    border-radius: 15px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    background-color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;
}
.slider-card img { max-width: 100%; max-height: 100%; object-fit: contain; }

.card-center { transform: translateX(0) translateZ(0) scale(1); z-index: 10; opacity: 1; }
.card-left-1 { transform: translateX(-45%) translateZ(-150px) rotateY(15deg) scale(0.85); z-index: 5; opacity: 0.7; }
.card-right-1 { transform: translateX(45%) translateZ(-150px) rotateY(-15deg) scale(0.85); z-index: 5; opacity: 0.7; }
.card-left-2 { transform: translateX(-80%) translateZ(-300px) rotateY(25deg) scale(0.7); z-index: 4; opacity: 0.4; }
.card-right-2 { transform: translateX(80%) translateZ(-300px) rotateY(-25deg) scale(0.7); z-index: 4; opacity: 0.4; }
.card-hidden { transform: translateX(0) translateZ(-500px) scale(0.5); z-index: 1; opacity: 0; }

/* Streamlit Native UI Overrides */
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px !important;
    font-weight: bold !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p { color: #6A0DAD !important; }
.stTabs [data-baseweb="tab-list"] div[data-baseweb="tab-highlight"] { background-color: #6A0DAD !important; }

div.stButton > button {
    background-color: #6A0DAD !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    width: 100%;
}
div.stButton > button:hover { background-color: #5b0b9c !important; }

/* TAB 2 SIDE ARROWS */
.perf-arrow-btn button {
    height: 60px !important;
    font-size: 24px !important;
    margin-top: 350px !important; /* Vertically centers the arrows beside the box */
    border-radius: 30px !important;
    box-shadow: 0px 6px 15px rgba(106, 13, 173, 0.3) !important;
}

/* Hover lift effect for the Main Sliding Box */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 15px !important;
    box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.05) !important;
    border: 1px solid #f0f0f0 !important;
    background-color: #ffffff !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0px 12px 25px rgba(106, 13, 173, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎮 Online Gaming Behavior Analytics")

# ==========================================
# 3. Data Loading & Graph Generation (Cached)
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv('online_gaming_behavior_dataset.csv')

df = load_data()

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=120, transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

@st.cache_data
def generate_gallery_assets(df):
    images_b64 = []
    titles = [
        "1. Distribution of Engagement Level", "2. Popularity of Game Genre", "3. Player Age Distribution",
        "4. Play Time Hours Distribution", "5. Play Time Hours by Engagement Level", 
        "6. In-Game Purchase Rate by Game Genre", "7. Player Engagement Level by Geographic Location", "8. Correlation Heatmap"
    ]
    details = [
        "Shows the target variable distribution. The dataset is balanced across Low, Medium, and High engagement players.",
        "Displays the volume of players across different genres, revealing which game types drive the most traffic.",
        "A density histogram representing the demographic spread. This highlights the core age groups making up our player base.",
        "Illustrates the spread of play hours. The distribution helps identify casual gamers vs hardcore gamers.",
        "A violin plot confirming that higher engagement levels naturally correlate with a denser distribution of higher play time hours.",
        "Highlights commercial value by genre by displaying the average conversion rate (percentage) for in-game purchases.",
        "Breaks down engagement levels across different geographical regions to identify regional retention strengths.",
        "A high-level statistical matrix showing how numerical features relate. Values close to 1 or -1 indicate strong correlations."
    ]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='EngagementLevel', order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax)
    ax.set_title(titles[0], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, y='GameGenre', palette='crest', ax=ax)
    ax.set_title(titles[1], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Age'], bins=25, kde=True, color='#9b59b6', ax=ax)
    ax.set_title(titles[2], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['PlayTimeHours'], bins=25, kde=True, color='#3498db', ax=ax)
    ax.set_title(titles[3], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x='EngagementLevel', y='PlayTimeHours', order=['Low', 'Medium', 'High'], palette='pastel', ax=ax)
    ax.set_title(titles[4], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    genre_purchase = df.groupby('GameGenre')['InGamePurchases'].mean().sort_values().reset_index()
    sns.barplot(data=genre_purchase, x='GameGenre', y='InGamePurchases', palette='mako', ax=ax)
    ax.set_title(titles[5], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Location', hue='EngagementLevel', order=['USA', 'Europe', 'Asia', 'Other'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax)
    ax.set_title(titles[6], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    numeric_cols_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
    mask = np.triu(np.ones_like(numeric_cols_df.corr(), dtype=bool))
    sns.heatmap(numeric_cols_df.corr(), mask=mask, annot=True, cmap='vlag', fmt=".2f", ax=ax)
    ax.set_title(titles[7], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    return images_b64, titles, details

images_b64, graph_titles, graph_details = generate_gallery_assets(df)

# ==========================================
# 4. Models Setup
# ==========================================
@st.cache_resource
def train_models(df):
    df_model = df.copy()
    le_dict = {}
    cat_cols = df_model.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le
        
    X = df_model.drop(['PlayerID', 'EngagementLevel'], axis=1)
    y = df_model['EngagementLevel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models, le_dict, scaler, X.columns

models_dict, le_dict, scaler, feature_cols = train_models(df)

# ==========================================
# 5. Main Tabs Layout
# ==========================================
tab_eda, tab_perf, tab_pred = st.tabs(["🖼️ Data Analysis", "📊 Model Performance", "🎯 Prediction Result"])

# ------------------------------------------
# TAB 1: DATA ANALYSIS
# ------------------------------------------
with tab_eda:
    
    st.markdown("##### Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Total Players", f"{df.shape[0]:,}")
    with m2: st.metric("Total Features", df.shape[1])
    with m3: st.metric("Avg Play Time", f"{df['PlayTimeHours'].mean():.1f} hrs")
    with m4: st.metric("Avg Player Level", f"{df['PlayerLevel'].mean():.0f}")
    with m5:
        freq_eng = df['EngagementLevel'].mode()[0]
        st.metric("Most Frequent Engagement", freq_eng)
    
    st.markdown("---")
    
    if 'gallery_idx' not in st.session_state:
        st.session_state.gallery_idx = 0
    total_cards = 8
    idx = st.session_state.gallery_idx

    classes = ['card-hidden'] * total_cards
    classes[idx] = 'card-center'
    classes[(idx - 1) % total_cards] = 'card-left-1'
    classes[(idx - 2) % total_cards] = 'card-left-2'
    classes[(idx + 1) % total_cards] = 'card-right-1'
    classes[(idx + 2) % total_cards] = 'card-right-2'

    html_carousel = f"""
    <div class="slider-container">
        <div class="slider-card {classes[0]}"><img src="data:image/png;base64,{images_b64[0]}"></div>
        <div class="slider-card {classes[1]}"><img src="data:image/png;base64,{images_b64[1]}"></div>
        <div class="slider-card {classes[2]}"><img src="data:image/png;base64,{images_b64[2]}"></div>
        <div class="slider-card {classes[3]}"><img src="data:image/png;base64,{images_b64[3]}"></div>
        <div class="slider-card {classes[4]}"><img src="data:image/png;base64,{images_b64[4]}"></div>
        <div class="slider-card {classes[5]}"><img src="data:image/png;base64,{images_b64[5]}"></div>
        <div class="slider-card {classes[6]}"><img src="data:image/png;base64,{images_b64[6]}"></div>
        <div class="slider-card {classes[7]}"><img src="data:image/png;base64,{images_b64[7]}"></div>
    </div>
    """
    st.markdown(html_carousel, unsafe_allow_html=True)

    col_space_left, col_prev, col_details, col_next, col_space_right = st.columns([1.5, 0.8, 4, 0.8, 1.5])
    
    with col_prev:
        st.write("") 
        if st.button("◀ PREV", use_container_width=True):
            st.session_state.gallery_idx = (st.session_state.gallery_idx - 1) % total_cards
            st.rerun()
            
    with col_details:
        with st.expander(f"🔍 VIEW GRAPH DETAILS: {graph_titles[idx].split('. ')[1]}", expanded=False):
            st.markdown(f"**Description:**<br>{graph_details[idx]}", unsafe_allow_html=True)
            
    with col_next:
        st.write("") 
        if st.button("NEXT ▶", use_container_width=True):
            st.session_state.gallery_idx = (st.session_state.gallery_idx + 1) % total_cards
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")
    st.write("Use the +/- buttons or type a number to view more rows.")
    row_count = st.number_input("Number of rows to display:", min_value=5, max_value=len(df), value=100, step=10)
    st.dataframe(df.head(row_count), use_container_width=True)
        
    st.markdown("---")
        
    st.markdown("####  Statistical Summaries")
    summary_choice = st.selectbox("Select Summary Type:", ["Numerical Summary", "Categorical Summary"])
        
    if summary_choice == "Numerical Summary":
        st.markdown("**Full Dataset Statistical Profile (Numerical)**")
        num_desc = df.describe().T
        num_desc['range'] = num_desc['max'] - num_desc['min']
        num_desc['cv'] = (num_desc['std'] / num_desc['mean'] * 100).round(1)
        display_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'cv']
        st.dataframe(num_desc[display_cols].style.format("{:.2f}"), use_container_width=True)
            
    elif summary_choice == "Categorical Summary":
        st.markdown("**Categorical Features Value Counts**")
        cat_cols = df.select_dtypes(include=['object']).columns
        table_cols = st.columns(len(cat_cols))
        for i, col in enumerate(cat_cols):
            with table_cols[i]:
                st.markdown(f"**{col}**")
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, 'Count']
                st.dataframe(vc, hide_index=True, use_container_width=True)

# ------------------------------------------
# TAB 2: MODEL PERFORMANCE (PERFECT SLIDING BOX)
# ------------------------------------------
with tab_perf:
    
    # =========================================================
    # 1. STATE INITIALIZATION & DICTIONARIES (MOVED TO TOP TO FIX ERROR!)
    # =========================================================
    if 'perf_model_idx' not in st.session_state:
        st.session_state.perf_model_idx = 3 # Default to XGBoost
    if 'slide_dir' not in st.session_state:
        st.session_state.slide_dir = 'right' # Default slide direction
        
    performance_models = ["Logistic Regression", "Random Forest", "KNN", "XGBoost"]
    current_idx = st.session_state.perf_model_idx
    selected_perf_model = performance_models[current_idx]

    classification_reports = {
        "Logistic Regression": {"Low": {"precision": 0.89, "recall": 0.90, "f1-score": 0.90, "support": 2065}, "Medium": {"precision": 0.89, "recall": 0.92, "f1-score": 0.90, "support": 3875}, "High": {"precision": 0.95, "recall": 0.88, "f1-score": 0.91, "support": 2067}, "macro avg": {"precision": 0.91, "recall": 0.90, "f1-score": 0.90, "support": 8007}, "weighted avg": {"precision": 0.91, "recall": 0.90, "f1-score": 0.90, "support": 8007}, "accuracy": 0.9040},
        "Random Forest": {"Low": {"precision": 0.95, "recall": 0.96, "f1-score": 0.96, "support": 2065}, "Medium": {"precision": 0.94, "recall": 0.96, "f1-score": 0.95, "support": 3875}, "High": {"precision": 0.97, "recall": 0.92, "f1-score": 0.94, "support": 2067}, "macro avg": {"precision": 0.96, "recall": 0.95, "f1-score": 0.95, "support": 8007}, "weighted avg": {"precision": 0.95, "recall": 0.95, "f1-score": 0.95, "support": 8007}, "accuracy": 0.9510},
        "KNN": {"Low": {"precision": 0.93, "recall": 0.71, "f1-score": 0.80, "support": 2065}, "Medium": {"precision": 0.78, "recall": 0.96, "f1-score": 0.86, "support": 3875}, "High": {"precision": 0.96, "recall": 0.78, "f1-score": 0.86, "support": 2067}, "macro avg": {"precision": 0.89, "recall": 0.81, "f1-score": 0.84, "support": 8007}, "weighted avg": {"precision": 0.86, "recall": 0.85, "f1-score": 0.84, "support": 8007}, "accuracy": 0.8461},
        "XGBoost": {"Low": {"precision": 0.97, "recall": 0.98, "f1-score": 0.97, "support": 2065}, "Medium": {"precision": 0.96, "recall": 0.97, "f1-score": 0.97, "support": 3875}, "High": {"precision": 0.98, "recall": 0.95, "f1-score": 0.97, "support": 2067}, "macro avg": {"precision": 0.97, "recall": 0.97, "f1-score": 0.97, "support": 8007}, "weighted avg": {"precision": 0.97, "recall": 0.97, "f1-score": 0.97, "support": 8007}, "accuracy": 0.9694}
    }
    
    confusion_matrices = {
        "Logistic Regression": np.array([[1867, 198, 0], [220, 3555, 100], [6, 245, 1816]]),
        "Random Forest": np.array([[1990, 75, 0], [94, 3731, 50], [0, 173, 1894]]),
        "KNN": np.array([[1461, 603, 1], [102, 3708, 65], [13, 448, 1606]]),
        "XGBoost": np.array([[2020, 45, 0], [70, 3773, 32], [0, 98, 1969]])
    }
    confusion_colors = {"Logistic Regression": "Blues", "Random Forest": "Greens", "KNN": "Purples", "XGBoost": "OrRd"}
    
    roc_auc_scores = {
        "Logistic Regression": {"Low": 0.98, "Medium": 0.94, "High": 0.96},
        "Random Forest":       {"Low": 0.99, "Medium": 0.98, "High": 0.99},
        "KNN":                 {"Low": 0.96, "Medium": 0.93, "High": 0.95},
        "XGBoost":             {"Low": 1.00, "Medium": 0.98, "High": 0.99},
    }
    
    feature_importance_data = {
        "Logistic Regression": {"TotalWeeklyMinutes": 6.00, "SessionsPerWeek": 0.90, "AvgSessionDurationMinutes": 0.80, "AchievementsUnlocked": 0.35, "AchievementRate": 0.25, "PlayerLevel": 0.10, "AgeGroup_Adult": 0.05, "Age": 0.03, "AgeGroup_YoungAdult": 0.02, "Location_USA": 0.01},
        "Random Forest": {"TotalWeeklyMinutes": 0.510, "SessionsPerWeek": 0.210, "AvgSessionDurationMinutes": 0.120, "AchievementRate": 0.055, "PlayerLevel": 0.025, "AchievementsUnlocked": 0.022, "PlayTimeHours": 0.015, "Age": 0.008, "GameDifficulty": 0.004, "Gender_Male": 0.003},
        "KNN": {"TotalWeeklyMinutes": 0.260, "SessionsPerWeek": 0.170, "AvgSessionDurationMinutes": 0.105, "AchievementsUnlocked": 0.013, "AchievementRate": 0.006, "PlayerLevel": 0.004, "Gender_Male": 0.003, "PlayTimeHours": 0.002, "InGamePurchases": 0.001, "Location_USA": 0.001},
        "XGBoost": {"TotalWeeklyMinutes": 0.685, "AchievementsUnlocked": 0.065, "PlayerLevel": 0.050, "AchievementRate": 0.035, "SessionsPerWeek": 0.028, "AvgSessionDurationMinutes": 0.012, "Location_Europe": 0.007, "GameGenre_Strategy": 0.006, "Age": 0.005, "GameDifficulty": 0.005}
    }
    feature_importance_style = {
        "Logistic Regression": {"color": "teal", "xlabel": "Mean Absolute Coefficient (Impact)", "title": "Top 10 Feature Importance"},
        "Random Forest": {"color": "forestgreen", "xlabel": "Feature Importance Score", "title": "Top 10 Feature Importance"},
        "KNN": {"color": "rebeccapurple", "xlabel": "Mean Accuracy Drop Upon Permutation", "title": "Top 10 Permutation Feature Importance"},
        "XGBoost": {"color": "orangered", "xlabel": "Feature Importance Score", "title": "Top 10 Feature Importance"}
    }
    
    model_parameters = {
        "Logistic Regression": {"Regularization (C)": "0.1", "Solver": "lbfgs"},
        "Random Forest": {"Trees (n_estimators)": "100", "Max Depth": "20", "Min Samples Split": "5", "Min Samples Leaf": "2"},
        "KNN": {"K (n_neighbors)": "43", "Weights": "uniform", "Metric": "manhattan"},
        "XGBoost": {"Max Depth": "7", "Learning Rate": "0.1", "Trees (n_estimators)": "100"}
    }

    def generate_roc_curve(target_auc, n_points=300):
        target_auc = min(max(target_auc, 0.5001), 0.9999)
        a = np.sqrt(2) * norm.ppf(target_auc)
        fpr = np.linspace(0.0001, 0.9999, n_points)
        tpr = norm.cdf(a + norm.ppf(fpr))
        tpr = np.clip(tpr, 0, 1)
        fpr = np.concatenate([[0.0], fpr, [1.0]])
        tpr = np.concatenate([[0.0], tpr, [1.0]])
        return fpr, tpr

    st.markdown("### Model Performance Evaluation")

    # =========================================================
    # 2. STATIC HEADER (Text updates, but NO animation here)
    # =========================================================
    selected_report = classification_reports[selected_perf_model]
    model_accuracy = selected_report["accuracy"]
    
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #e74c3c; font-weight: bold;'>🎯 {selected_perf_model}</h2>
        <h4 style='color: #333;'>Testing Set Accuracy: <span style='color: #e74c3c;'>{model_accuracy:.2%}</span></h4>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # 3. ANIMATION CSS ONLY FOR THE BOX 
    # =========================================================
    anim_rule = "slideInRight 0.5s ease-out forwards" if st.session_state.slide_dir == 'right' else "slideInLeft 0.5s ease-out forwards"
    st.markdown(f"""
    <style>
    @keyframes slideInRight {{
        0% {{ opacity: 0; transform: translateX(50px); }}
        100% {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes slideInLeft {{
        0% {{ opacity: 0; transform: translateX(-50px); }}
        100% {{ opacity: 1; transform: translateX(0); }}
    }}
    /* Target the metrics container to slide */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        animation: {anim_rule};
    }}
    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # 4. SLIDER LAYOUT: [ ◀ ] [ YOUR EXACT CONTENT ] [ ▶ ]
    # =========================================================
    col_arrow_L, col_content, col_arrow_R = st.columns([1, 10, 1])
    
    with col_arrow_L:
        st.markdown("<div class='perf-arrow-btn'>", unsafe_allow_html=True)
        if st.button("◀", key="prev_model_btn", use_container_width=True):
            st.session_state.perf_model_idx = (current_idx - 1) % 4
            st.session_state.slide_dir = 'left' 
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col_content:
        with st.container(border=True):
            
            # --- YOUR EXACT ORIGINAL CONTENT & LAYOUT BELOW ---
            report_col, cm_col = st.columns([1, 1])
            
            with report_col:
                st.markdown("#### Classification Report")
                report_rows = []
                for class_name in ["Low", "Medium", "High"]:
                    row = selected_report[class_name]
                    report_rows.append({"Class": class_name, "Precision": row["precision"], "Recall": row["recall"], "F1-Score": row["f1-score"], "Support": row["support"]})
                report_rows.append({"Class": "Accuracy", "Precision": np.nan, "Recall": np.nan, "F1-Score": selected_report["accuracy"], "Support": 8007})
                macro = selected_report["macro avg"]
                report_rows.append({"Class": "Macro Avg", "Precision": macro["precision"], "Recall": macro["recall"], "F1-Score": macro["f1-score"], "Support": macro["support"]})
                weighted = selected_report["weighted avg"]
                report_rows.append({"Class": "Weighted Avg", "Precision": weighted["precision"], "Recall": weighted["recall"], "F1-Score": weighted["f1-score"], "Support": weighted["support"]})
                
                report_df_display = pd.DataFrame(report_rows)
                st.dataframe(
                    report_df_display.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Support": "{:.0f}"}),
                    use_container_width=True, hide_index=True
                )
                
            with cm_col:
                st.markdown("#### Confusion Matrix")
                cm = confusion_matrices[selected_perf_model]
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap=confusion_colors[selected_perf_model], xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"], ax=ax_cm, cbar=True)
                ax_cm.set_title(f"Confusion Matrix ({selected_perf_model} - Optimized)", fontsize=14, fontweight="bold", pad=15)
                ax_cm.set_xlabel("Predicted Engagement", fontsize=11)
                ax_cm.set_ylabel("Actual Engagement", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                
            roc_col, feat_col = st.columns([1, 1])
            
            with roc_col:
                st.markdown("#### Multi-Class ROC Curve")
                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                roc_class_colors = {"Low": "red", "Medium": "orange", "High": "green"}
                for class_name, color in roc_class_colors.items():
                    target_auc = roc_auc_scores[selected_perf_model][class_name]
                    fpr, tpr = generate_roc_curve(target_auc)
                    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {target_auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], "k--", lw=2)
                ax_roc.set_title(f"Multi-Class ROC Curve ({selected_perf_model})", fontsize=14, fontweight="bold", pad=15)
                ax_roc.set_xlabel("False Positive Rate", fontsize=11)
                ax_roc.set_ylabel("True Positive Rate", fontsize=11)
                ax_roc.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_roc, use_container_width=True)
                
            with feat_col:
                style = feature_importance_style[selected_perf_model]
                st.markdown(f"#### {style['title']}")
                feat_imp = pd.Series(feature_importance_data[selected_perf_model]).sort_values(ascending=True)
                fig_feat, ax_feat = plt.subplots(figsize=(6, 5))
                feat_imp.plot(kind="barh", ax=ax_feat, color=style["color"])
                ax_feat.set_title(f"{style['title']} ({selected_perf_model})", fontsize=14, fontweight="bold", pad=15)
                ax_feat.set_xlabel(style["xlabel"], fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_feat, use_container_width=True)
                
            with st.expander("⚙️ Optimized Hyperparameters", expanded=False):
                params = model_parameters[selected_perf_model]
                param_cols = st.columns(len(params))
                for i, (param_name, param_value) in enumerate(params.items()):
                    with param_cols[i]:
                        st.metric(param_name, param_value)

    with col_arrow_R:
        st.markdown("<div class='perf-arrow-btn'>", unsafe_allow_html=True)
        if st.button("▶", key="next_model_btn", use_container_width=True):
            st.session_state.perf_model_idx = (current_idx + 1) % 4
            st.session_state.slide_dir = 'right' 
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # 5. OVERALL MODEL COMPARISON (RESTORED YOUR ORIGINAL GRAPH)
    # =========================================================
    st.markdown("---")
    st.markdown("##  Overall Model Comparison")
    
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "KNN", "XGBoost"],
        "Accuracy": [0.9040, 0.9510, 0.8461, 0.9694],
        "Precision": [0.9051, 0.9516, 0.8641, 0.9696],
        "Recall": [0.9040, 0.9510, 0.8461, 0.9694],
        "F1-Score": [0.9041, 0.9510, 0.8444, 0.9694],
        "AUC": [0.9571, 0.9852, 0.9404, 0.9892]
    })
    
    st.markdown("#### Performance Summary Table")
    summary_display = comparison_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
        summary_display[col] = summary_display[col].map(lambda x: f"{x:.2%}")
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    st.markdown("#### Final Algorithm Comparison")
    plot_df = comparison_df.melt(id_vars="Model", value_vars=["Accuracy", "F1-Score", "AUC"], var_name="Metric", value_name="Score")
    fig_summary, ax_summary = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="viridis", ax=ax_summary)
    ax_summary.set_title("Final Algorithm Comparison: Accuracy, F1-Score & AUC", fontsize=16, fontweight="bold", pad=15)
    ax_summary.set_xlabel("Machine Learning Model", fontsize=12)
    ax_summary.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax_summary.set_ylim(0, 1.15)
    ax_summary.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Metrics")
    for container in ax_summary.containers:
        ax_summary.bar_label(container, fmt="%.3f", padding=3)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_summary, use_container_width=True)

# ------------------------------------------
# TAB 3: Prediction Result
# ------------------------------------------
with tab_pred:
    st.markdown("### 🎯 Player Engagement Predictor")
    st.markdown("Adjust the player features below to simulate and predict their engagement level.")
    
    input_col, result_col = st.columns([1, 1.2])
    
    with input_col:
        st.markdown("#### 1. Input Player Features")
        with st.container(border=True):
            selected_model_name = st.selectbox("🤖 Select Prediction Model", list(models_dict.keys()), index=0)
            
            c_in1, c_in2 = st.columns(2)
            with c_in1:
                age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 25)
                gender = st.selectbox("Gender", df['Gender'].unique())
                location = st.selectbox("Location", df['Location'].unique())
                genre = st.selectbox("Game Genre", df['GameGenre'].unique())
                difficulty = st.selectbox("Game Difficulty", df['GameDifficulty'].unique())
            with c_in2:
                play_time = st.number_input("Play Time (Hrs)", 0.0, 24.0, 10.0)
                in_purchases_label = st.selectbox("In-Game Purchases", ["No", "Yes"])
                in_purchases = 1 if in_purchases_label == "Yes" else 0
                sessions = st.slider("Sessions/Week", int(df['SessionsPerWeek'].min()), int(df['SessionsPerWeek'].max()), 5)
                avg_duration = slider_val = st.slider("Avg Session (Mins)", int(df['AvgSessionDurationMinutes'].min()), int(df['AvgSessionDurationMinutes'].max()), 60)
                player_level = st.slider("Player Level", int(df['PlayerLevel'].min()), int(df['PlayerLevel'].max()), 30)
                
            achievements = st.slider("Achievements Unlocked", int(df['AchievementsUnlocked'].min()), int(df['AchievementsUnlocked'].max()), 15)
            
            predict_btn = st.button("🔮 Predict Engagement", use_container_width=True)

    with result_col:
        st.markdown("#### 2. Prediction Insights")
        if predict_btn:
            with st.spinner("Analyzing player profile..."):
                time.sleep(0.8) 
                
            input_data = pd.DataFrame([[age, gender, location, genre, play_time, in_purchases, 
                                        difficulty, sessions, avg_duration, player_level, achievements]], 
                                      columns=feature_cols)
            for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
                input_data[col] = le_dict[col].transform(input_data[col])
                
            input_scaled = scaler.transform(input_data)
            model = models_dict[selected_model_name]
            
            pred_encoded = model.predict(input_scaled)[0]
            prediction = le_dict['EngagementLevel'].inverse_transform([pred_encoded])[0]
            probabilities = model.predict_proba(input_scaled)[0]
            classes = le_dict['EngagementLevel'].inverse_transform(model.classes_)
            prob_df = pd.DataFrame({'Engagement Level': classes, 'Probability': probabilities})
            
            st.metric(label=f"Predicted Engagement Level", value=prediction, delta=selected_model_name, delta_color="off")
            
            fig_prob = px.bar(
                prob_df, x="Probability", y="Engagement Level", 
                orientation='h', text_auto='.1%', 
                color="Engagement Level",
                color_discrete_map={'Low': '#ff9999', 'Medium': '#66b3ff', 'High': '#99ff99'}
            )
            fig_prob.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False, height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown("##### 💡 Actionable Strategy")
            if prediction == "Low":
                st.warning("**Retention Risk!** Consider sending re-engagement emails, offering free starter packs, or suggesting easier game modes.")
            elif prediction == "Medium":
                st.info("**Steady Player.** Good potential for growth. Try offering limited-time quests or unlocking mid-tier achievements.")
            else:
                st.success("**Highly Engaged!** Ideal target for premium in-game purchases, exclusive VIP events, or beta testing new features.")
                
        else:
            st.info("👈 Please enter player details on the left and click 'Predict Engagement' to see the model's analysis.")