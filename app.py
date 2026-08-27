import streamlit as st
import streamlit.components.v1 as components
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
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100, transparent=True)
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
        "Shows the target variable distribution. The dataset is balanced across Low, Medium, and High engagement players, providing a solid baseline for our ML predictions.",
        "Displays the volume of players across different genres (Sports, Action, Strategy, etc.), revealing which game types drive the most traffic.",
        "A density histogram representing the demographic spread. This highlights the core age groups making up our player base.",
        "Illustrates the spread of play hours. The distribution helps identify the threshold between casual gamers and hardcore gamers.",
        "A violin plot confirming that higher engagement levels naturally correlate with a denser distribution of higher play time hours.",
        "Highlights commercial value by genre. It displays the average conversion rate (percentage) for in-game purchases.",
        "Breaks down engagement levels across different geographical regions, useful for identifying regional retention strengths.",
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
# 4. Models Setup & Data Dictionaries
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

# Notebook Data Dictionaries (Performance Metrics)
perf_models_list = ["Logistic Regression", "Random Forest", "KNN", "XGBoost"]

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
    "KNN": {"color": "rebeccapurple", "xlabel": "Mean Accuracy Drop Upon Perm", "title": "Top 10 Permutation Importance"},
    "XGBoost": {"color": "orangered", "xlabel": "Feature Importance Score", "title": "Top 10 Feature Importance"}
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


# ----------------------------------------------------
# 4.1 HTML/CSS Widget Generator for the EDA 3D Slider (With Flip effect)
# ----------------------------------------------------
@st.cache_data
def generate_eda_slider_html(images_b64, titles, details):
    slides_html = ""
    
    for i in range(len(images_b64)):
        slides_html += f"""
        <div class="slide" onclick="flipCard({i})">
            <div class="card-inner">
                <div class="card-front">
                    <img src="data:image/png;base64,{images_b64[i]}">
                    <div class="hint-text">🖱️ Click to view details</div>
                </div>
                <div class="card-back">
                    <h3>{titles[i]}</h3>
                    <p>{details[i]}</p>
                    <div class="hint-text">🖱️ Click to flip back</div>
                </div>
            </div>
        </div>
        """

    widget_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;800&display=swap');
      body {{ margin: 0; padding: 0; font-family: 'Source Sans Pro', sans-serif; overflow: hidden; background: transparent; }}
      
      .slider-container {{ 
          position: relative; width: 100%; height: 520px; 
          display: flex; justify-content: center; align-items: center; 
          perspective: 1200px; overflow: hidden;
      }}
      
      .slide {{
          position: absolute; width: 700px; height: 420px;
          transition: transform 0.6s cubic-bezier(0.25, 0.8, 0.25, 1), opacity 0.6s ease;
          border-radius: 15px; background: transparent;
      }}
      
      /* Active state */
      .slide.active {{ transform: translateX(0) scale(1) translateZ(0); opacity: 1; z-index: 10; cursor: pointer; }}
      /* Adjacent models peek out with 3D rotation */
      .slide.left-1 {{ transform: translateX(-65%) scale(0.8) translateZ(-150px) rotateY(15deg); opacity: 0.5; z-index: 5; pointer-events: none; }}
      .slide.right-1 {{ transform: translateX(65%) scale(0.8) translateZ(-150px) rotateY(-15deg); opacity: 0.5; z-index: 5; pointer-events: none; }}
      .slide.hidden {{ transform: translateX(0) scale(0.5) translateZ(-400px); opacity: 0; z-index: 1; pointer-events: none; }}
      
      /* Flip Card Mechanics */
      .card-inner {{
          position: relative; width: 100%; height: 100%;
          text-align: center; transition: transform 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          transform-style: preserve-3d; border-radius: 15px;
          box-shadow: 0 15px 35px rgba(0,0,0,0.15);
      }}
      
      .slide.active.flipped .card-inner {{
          transform: rotateY(180deg);
      }}
      
      .card-front, .card-back {{
          position: absolute; width: 100%; height: 100%;
          -webkit-backface-visibility: hidden; backface-visibility: hidden;
          border-radius: 15px; background: white;
          display: flex; justify-content: center; align-items: center;
          padding: 15px; box-sizing: border-box;
      }}
      
      .card-front img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
      
      .card-back {{
          transform: rotateY(180deg);
          flex-direction: column; background: #fafafa;
          border: 4px solid #6A0DAD;
          padding: 40px; text-align: left;
      }}
      .card-back h3 {{ color: #6A0DAD; margin-top: 0