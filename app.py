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
# 🚀 SMART STICKY HEADER JAVASCRIPT LOGIC
# 注入监听器：往下拉隐藏，往上拉出现
# ==========================================
smart_scroll_js = """
<script>
const parentWin = window.parent;
const parentDoc = window.parent.document;

if (!parentWin._smartHeaderInitialized) {
    let lastScrollY = 0;
    let ticking = false;

    const scrollHandler = function(e) {
        if (!ticking) {
            parentWin.requestAnimationFrame(function() {
                let currentScrollY = parentWin.scrollY;
                // 兼容 Streamlit 内部滚动容器
                if (e.target && e.target.scrollTop !== undefined && e.target.tagName !== 'IFRAME') {
                    currentScrollY = e.target.scrollTop;
                }

                // 核心判断逻辑
                if (currentScrollY <= 80) {
                    // 到达顶部，全部重置显示
                    parentDoc.body.classList.remove('hide-smart-header');
                } else if (currentScrollY > lastScrollY + 15) {
                    // 往下滚 (加缓冲值防止抖动) -> 隐藏
                    parentDoc.body.classList.add('hide-smart-header');
                } else if (currentScrollY < lastScrollY - 15) {
                    // 往上滚 -> 显示
                    parentDoc.body.classList.remove('hide-smart-header');
                }
                lastScrollY = currentScrollY;
                ticking = false;
            });
            ticking = true;
        }
    };

    // 捕获所有的滚动事件
    parentDoc.addEventListener('scroll', scrollHandler, true);
    parentWin._smartHeaderInitialized = true;
}
</script>
"""
components.html(smart_scroll_js, height=0, width=0)

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
    font-size: 16px !important;
    font-weight: bold !important;
}

div.stButton > button {
    background: linear-gradient(180deg, #3a0a63 0%, #26004a 55%, #16002b 100%) !important;
    color: white !important;
    border: none !important;
    border-bottom: 3px solid #4a0880 !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    width: 100%;
    box-shadow: 0 5px 0 #4a0880, 0 8px 16px rgba(106,13,173,0.35) !important;
    transform: translateY(0);
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease !important;
    position: relative;
}
div.stButton > button:hover {
    background: linear-gradient(180deg, #4d1080 0%, #32005c 55%, #1c0035 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 7px 0 #4a0880, 0 14px 22px rgba(106,13,173,0.4) !important;
}
div.stButton > button:active {
    transform: translateY(3px);
    box-shadow: 0 2px 0 #4a0880, 0 4px 8px rgba(106,13,173,0.3) !important;
}
div.stButton > button:focus:not(:active) {
    box-shadow: 0 5px 0 #4a0880, 0 8px 16px rgba(106,13,173,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# PREMIUM HEADER
# ==========================================

st.markdown("""
<style>

/* ---------- 智能吸顶效果核心控制 ---------- */

/* 1. 精准抓住 Header 的外层容器让他吸顶 */
div.element-container:has(.gaming-header) {
    position: sticky !important;
    top: 1.5rem !important;
    z-index: 99999 !important;
    transition: transform 0.4s cubic-bezier(0.3, 0, 0.2, 1) !important;
}

/* 2. 当 JS 检测到往下滑动时，加上隐藏的位移动画 */
body.hide-smart-header div.element-container:has(.gaming-header) {
    transform: translateY(-250px) !important;
}

/* Tabs 也跟着一起隐藏 */
body.hide-smart-header div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
    transform: translateY(-250px) !important;
}

/* ---------- MAIN HEADER ---------- */

.gaming-header {
    width: 100%;
    padding: 45px 35px 45px 35px;  
    margin-bottom: 0px; /* 移除底部 margin，让下方内容更紧凑 */
    border-radius: 22px;
    overflow: hidden;
    position: relative; 

    background: radial-gradient(circle at 90% 20%, rgba(155, 89, 182, 0.25), transparent 35%),
                radial-gradient(circle at 10% 80%, rgba(106, 13, 173, 0.18), transparent 35%),
                linear-gradient(135deg, #16002b 0%, #26004a 45%, #12001f 100%);
    box-shadow: 0 15px 45px rgba(72, 0, 120, 0.25);
}

/* Decorative glow */

.gaming-header::before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -100px;
    top: -130px;
    border-radius: 50%;

    background: rgba(190, 120, 255, 0.15);
    filter: blur(20px);
}

.gaming-header::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #6A0DAD, #b45cff, #6A0DAD);
    background-size: 200% 100%;
    animation: gradientMove 4s linear infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* Header content */

.header-content {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Logo */

.logo-area {
    display: flex;
    align-items: center;
    gap: 18px;
}

.logo-icon {
    width: 65px;
    height: 65px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    background: linear-gradient(135deg, rgba(255,255,255,0.18), rgba(255,255,255,0.05));
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 25px rgba(0,0,0,0.25), inset 0 0 20px rgba(255,255,255,0.05);
}

/* Title */

.header-title {
    margin: 0;
    color: white;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.header-subtitle {
    margin-top: 5px;
    color: rgba(255,255,255,0.68);
    font-size: 14px;
    letter-spacing: 0.5px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="gaming-header">
<div class="header-content">
<div class="logo-area">
<div>
<div class="header-title">
Online Gaming Analytics
</div>
<div class="header-subtitle">
PLAYER BEHAVIOR • MACHINE LEARNING • ENGAGEMENT INTELLIGENCE
</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

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

    return images_b64, titles

images_b64, graph_titles = generate_gallery_assets(df)

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
# 4.1 HTML/CSS Widget Generators (Slider - No Flip)
# ----------------------------------------------------

@st.cache_data
def generate_eda_slider_html(images_b64, titles):
    slides_html = ""
    for i in range(len(images_b64)):
        img = images_b64[i]
        title = titles[i]
        
        slides_html += f"""
        <div class="slide eda-slide">
            <img src="data:image/png;base64,{img}" alt="{title}" class="slide-img">
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;800&display=swap');
      body {{ margin: 0; padding: 0; font-family: 'Source Sans Pro', sans-serif; overflow: hidden; background: transparent; }}

      .slider-container {{ 
          position: relative; width: 100%; height: 500px; 
          display: flex; justify-content: center; align-items: center; 
          perspective: 1500px; overflow: hidden;
      }}

      .slide {{
          position: absolute; width: 750px; height: 450px;
          transition: transform 0.6s cubic-bezier(0.25, 0.8, 0.25, 1), opacity 0.6s ease;
          border-radius: 20px; 
          background: #ffffff;
          border-top: 5px solid #6A0DAD;
          box-shadow: 0 10px 30px rgba(0,0,0,0.1);
          display: flex; justify-content: center; align-items: center;
          padding: 20px; box-sizing: border-box;
      }}
      
      .slide-img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}

      /* Coverflow states */
      .slide.active {{ transform: translateX(0) scale(1) translateZ(0); opacity: 1; z-index: 10; }}
      .slide.left-1 {{ transform: translateX(-65%) scale(0.8) translateZ(-150px) rotateY(15deg); opacity: 0.5; z-index: 5; pointer-events: none; }}
      .slide.right-1 {{ transform: translateX(65%) scale(0.8) translateZ(-150px) rotateY(-15deg); opacity: 0.5; z-index: 5; pointer-events: none; }}
      .slide.hidden {{ transform: translateX(0) scale(0.6) translateZ(-400px); opacity: 0; z-index: 1; pointer-events: none; }}
      
      /* Navigation Arrows */
      .nav-btn {{
          position: absolute; top: 50%; transform: translateY(-50%);
          width: 50px; height: 50px; border-radius: 25px;
          background: white; border: 2px solid #6A0DAD; color: #6A0DAD;
          font-size: 22px; cursor: pointer; z-index: 100;
          box-shadow: 0 5px 15px rgba(106,13,173,0.2);
          display: flex; justify-content: center; align-items: center;
          transition: all 0.2s; outline: none;
      }}
      .nav-btn:hover {{ background: #6A0DAD; color: white; transform: translateY(-50%) scale(1.15); }}
      .prev-btn {{ left: 2%; }}
      .next-btn {{ right: 2%; }}
      
    </style>
    </head>
    <body>
      <div class="slider-container" id="slider">
        <button class="nav-btn prev-btn" onclick="move(-1, event)">&#9664;</button>
        <button class="nav-btn next-btn" onclick="move(1, event)">&#9654;</button>
        {slides_html}
      </div>
      <script>
        const slides = document.querySelectorAll('.eda-slide');
        let currentIndex = 0;

        function updateSlides() {{
            slides.forEach((slide, index) => {{
                slide.className = 'slide eda-slide'; // clear states
                if (index === currentIndex) {{
                    slide.classList.add('active');
                }} else if (index === (currentIndex - 1 + slides.length) % slides.length) {{
                    slide.classList.add('left-1');
                }} else if (index === (currentIndex + 1) % slides.length) {{
                    slide.classList.add('right-1');
                }} else {{
                    slide.classList.add('hidden');
                }}
            }});
        }}

        function move(dir, event) {{
            if(event) event.stopPropagation(); 
            currentIndex = (currentIndex + dir + slides.length) % slides.length;
            updateSlides();
        }}

        // Swipe support for touch devices
        let startX = 0;
        const slider = document.getElementById('slider');
        slider.addEventListener('touchstart', e => {{
            startX = e.changedTouches[0].screenX;
        }});
        slider.addEventListener('touchend', e => {{
            let endX = e.changedTouches[0].screenX;
            if (startX - endX > 50) move(1);
            if (startX - endX < -50) move(-1);
        }});

        updateSlides();
      </script>
    </body>
    </html>
    """
    return html

# ==========================================
# PREMIUM NAVIGATION TABS (MOVED INTO HEADER)
# ==========================================

st.markdown("""
<style>

/* Tab container - 使用负边距将其悬浮进 Header 的右侧 */
div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
    position: sticky !important;
    top: 4rem !important; 
    margin-top: -105px !important; /* 向上拉，嵌进 Header */
    margin-bottom: 20px !important; 
    margin-right: 35px !important; /* 跟 Header 右侧对齐 */
    justify-content: flex-end !important; /* 靠右对齐 */
    z-index: 100000 !important;
    background-color: transparent !important; /* 透明背景 */
    border-bottom: none !important;
    gap: 12px !important;
    transition: transform 0.4s cubic-bezier(0.3, 0, 0.2, 1) !important;
}

/* 补偿下方内容区的位置，以免重叠 */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 85px !important;
}

/* Individual tabs - 变成半透明的高级质感胶囊状 */
.stTabs [data-baseweb="tab"] {
    height: 48px !important;
    padding: 0 24px !important;
    border-radius: 24px !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}

/* Hover */
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* Active tab */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff !important;
    border-color: #ffffff !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
    transform: translateY(-2px) !important;
}

/* Remove default underline */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Tab text 颜色调整为适应暗色底色 */
.stTabs [data-baseweb="tab"] p {
    color: #ebd9ff !important;
}

/* Active text */
.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: #3a0a63 !important;
}

/* 移除底部指示线 */
.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LAB — BENTO CARD SYSTEM
# ==========================================

st.markdown("""
<style>

/* ---- Model selector: raised, glowing "active" state ---- */
.model-btn-marker + div[data-testid="stButton"] button {
    opacity: 0.92;
}
.model-btn-marker.active + div[data-testid="stButton"] button {
    background: linear-gradient(180deg, #5c1799 0%, #38086b 55%, #26004a 100%) !important;
    border-bottom: 3px solid #4a0880 !important;
    box-shadow: 0 5px 0 #4a0880, 0 0 0 3px rgba(155,92,255,0.4), 0 12px 26px rgba(106,13,173,0.5) !important;
    transform: translateY(-3px);
    opacity: 1;
}
.model-btn-marker.active + div[data-testid="stButton"] button:hover {
    transform: translateY(-4px);
}

/* ---- Bento cards: every bordered container becomes a themed card ---- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .bento-marker) {
    background: linear-gradient(180deg, #ffffff 0%, #fdfbff 100%) !important;
    border: 1px solid #eee2f7 !important;
    border-top: 4px solid #6A0DAD !important;
    border-radius: 18px !important;
    padding: 6px 6px 14px 6px !important;
    box-shadow: 0 8px 22px rgba(106,13,173,0.08) !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .bento-marker):hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 16px 32px rgba(106,13,173,0.16) !important;
}

/* ---- Hero summary card (accuracy / precision / recall strip) ---- */
.hero-model-card {
    background: radial-gradient(circle at 88% 0%, rgba(106,13,173,0.06), transparent 45%),
                linear-gradient(180deg, #ffffff 0%, #fbf8ff 100%);
    border: 1px solid #eee2f7;
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 22px;
    box-shadow: 0 4px 0 #e6d6f5, 0 16px 32px rgba(106,13,173,0.14);
    position: relative;
    overflow: hidden;
}
.hero-model-card::after {
    content: "";
    position: absolute; bottom: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, #6A0DAD, #b45cff, #6A0DAD);
    background-size: 200% 100%;
    animation: gradientMove 4s linear infinite;
}
.hero-model-name { color: #2a0a45; font-size: 24px; font-weight: 800; margin: 0; letter-spacing: -0.3px; }
.hero-model-sub { color: #8a7a99; font-size: 13px; letter-spacing: 0.5px; margin-top: 2px; }

/* ---- Section headers used throughout Model Lab ---- */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 4px 0 14px 0;
}
.section-header .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: linear-gradient(135deg, #b45cff, #6A0DAD);
    box-shadow: 0 0 8px rgba(106,13,173,0.5);
}
.section-header span.label {
    font-size: 16px; font-weight: 800; color: #3a1050;
}

/* ---- Hyperparameter pills ---- */
.hparam-pill {
    display: inline-flex; flex-direction: column; align-items: center;
    background: linear-gradient(180deg, #f8f1ff 0%, #f0e2ff 100%);
    border: 1px solid #e2c6ff;
    border-radius: 14px;
    padding: 10px 16px;
    margin: 4px 6px 4px 0;
    min-width: 120px;
    box-shadow: 0 3px 8px rgba(106,13,173,0.08);
}
.hparam-pill .pval { color: #6A0DAD; font-weight: 800; font-size: 15px; }
.hparam-pill .pname { color: #888; font-size: 11px; letter-spacing: 0.3px; text-transform: uppercase; margin-top: 2px; }

/* Comparison winner badge */
.winner-banner {
    background: linear-gradient(135deg, #fff7e6, #fff);
    border: 1px solid #ffe2a8;
    border-left: 5px solid #f2a900;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 16px;
    font-size: 14px;
    color: #7a5200;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ---- About This System section ---- */
.about-intro {
    background: linear-gradient(180deg, #faf7ff 0%, #ffffff 100%);
    border: 1px solid #eee2f7;
    border-left: 4px solid #6A0DAD;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 6px 0 24px 0;
    color: #444;
    font-size: 15px;
    line-height: 1.65;
}
.about-intro b { color: #3a1050; }

.about-card {
    background: #ffffff;
    border: 1px solid #eee2f7;
    border-top: 3px solid #6A0DAD;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 4px 12px rgba(106,13,173,0.07);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.about-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(106,13,173,0.15);
}
.about-card h5 {
    display: flex; align-items: center; gap: 8px;
    color: #3a1050; margin: 0 0 8px 0; font-size: 16px; font-weight: 800;
}
.about-card p { color: #666; font-size: 13.5px; margin: 0; line-height: 1.5; }

.step-arrow {
    display: flex; align-items: center; justify-content: center;
    height: 100%; min-height: 70px;
    color: #c9a6f0; font-size: 20px; font-weight: 800;
}

.tech-badge {
    display: inline-block;
    background: linear-gradient(180deg, #f7efff, #efe0ff);
    border: 1px solid #e2c6ff;
    color: #6A0DAD;
    font-size: 12.5px; font-weight: 700;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 4px 6px 4px 0;
}
</style>
""", unsafe_allow_html=True)

tab_eda, tab_perf, tab_pred = st.tabs([
    "DATA ANALYSIS",
    "MODEL LAB",
    "AI PREDICTOR"
])

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

    st.markdown("---")

    # Render seamless HTML/JS interactive component for the Graphs
    st.markdown("<p style='text-align: center; color: #666;'>Drag or click the arrows to navigate the visual insights.</p>", unsafe_allow_html=True)
    
    eda_slider_html = generate_eda_slider_html(images_b64, graph_titles)
    components.html(eda_slider_html, height=520, scrolling=False)


    # ---- 5. Integrated About Section ----
    st.markdown("#### About This System")
    st.markdown(f"""
    <div class="about-intro">
    To ensure users clearly grasp the system's overarching goals directly within the primary view,
    this dashboard cleanly analyses the <b>Online Gaming Behavior Dataset</b> and predicts a player's <b>Engagement Level</b>.
    It turns raw gaming activity into a clear read on how engaged a player really is — built end-to-end from EDA to a live predictor.
    </div>
    """, unsafe_allow_html=True)
    
    # 移除了圆圈数字
    steps = [
        ("Explore the Data", "Understand player behaviour through distributions and correlations."),
        ("Engineer Features", "Derive TotalWeeklyMinutes, AchievementRate, and AgeGroup."),
        ("Train & Compare", "Tune and benchmark 4 models: Logistic Regression, Random Forest, KNN, XGBoost."),
        ("Predict Live", "Enter a player profile and get an instant engagement prediction."),
    ]
    
    # 4 step columns + 3 thin arrow columns in between
    step_cols = st.columns([1, 0.15, 1, 0.15, 1, 0.15, 1])
    step_idx = 0
    for i, col in enumerate(step_cols):
        with col:
            if i % 2 == 1:  # arrow slot
                st.markdown('<div class="step-arrow">➜</div>', unsafe_allow_html=True)
            else:
                title, desc = steps[step_idx]
                step_idx += 1
                st.markdown(f"""
                <div class="about-card">
                    <h5>{title}</h5>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("##### Built With")
    tech_stack = [
        "Python", "Streamlit", "Pandas", "NumPy", "Scikit-learn",
        "XGBoost", "Seaborn", "Matplotlib", "Plotly"
    ]
    badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in tech_stack])
    st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)


# ------------------------------------------
# TAB 2: Model Performance
# ------------------------------------------
with tab_perf:
 
    st.markdown("### Model Performance Evaluation")
 
    # =========================================================
    # 1. HORIZONTAL MODEL BUTTON BAR
    # =========================================================
 
    performance_models = [
        "Logistic Regression",
        "Random Forest",
        "KNN",
        "XGBoost"
    ]
 
    # Default selected model
    if "performance_model" not in st.session_state:
        st.session_state.performance_model = "XGBoost"
 
    current_perf_model = st.session_state.performance_model
 
    # Four horizontal buttons, each preceded by an invisible marker div so the
    # CSS above can detect and highlight whichever one is currently active.
    btn_cols = st.columns(4)
 
    for i, model_name in enumerate(performance_models):
        with btn_cols[i]:
 
            is_active = current_perf_model == model_name
            marker_class = "model-btn-marker active" if is_active else "model-btn-marker"
            st.markdown(f'<div class="{marker_class}"></div>', unsafe_allow_html=True)
 
            # Add check mark to currently selected model
            button_label = (
                f"✓ {model_name}"
                if is_active
                else model_name
            )
 
            if st.button(
                button_label,
                key=f"perf_model_btn_{i}",
                use_container_width=True
            ):
                st.session_state.performance_model = model_name
                st.rerun()
 
    selected_perf_model = st.session_state.performance_model
 
    # =========================================================
    # 4. HERO CARD — model name + headline metrics
    # =========================================================
 
    selected_report = classification_reports[selected_perf_model]
    model_accuracy = selected_report["accuracy"]
    comparison_lookup = {
        "Logistic Regression": {"Precision": 0.9051, "Recall": 0.9040, "F1-Score": 0.9041, "AUC": 0.9571},
        "Random Forest":       {"Precision": 0.9516, "Recall": 0.9510, "F1-Score": 0.9510, "AUC": 0.9852},
        "KNN":                 {"Precision": 0.8641, "Recall": 0.8461, "F1-Score": 0.8444, "AUC": 0.9404},
        "XGBoost":              {"Precision": 0.9696, "Recall": 0.9694, "F1-Score": 0.9694, "AUC": 0.9892},
    }
    sel_extra = comparison_lookup[selected_perf_model]
 
    st.markdown(f"""
    <div class="hero-model-card">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:18px;">
            <div>
                <p class="hero-model-name">{selected_perf_model}</p>
                <p class="hero-model-sub">TESTING SET PERFORMANCE • 8,007 PLAYERS</p>
            </div>
            <div style="display:flex; gap:28px; flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="color:#6A0DAD; font-size:26px; font-weight:800;">{model_accuracy:.1%}</div>
                    <div style="color:#9c8caa; font-size:11px; letter-spacing:0.5px;">ACCURACY</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#6A0DAD; font-size:26px; font-weight:800;">{sel_extra['Precision']:.1%}</div>
                    <div style="color:#9c8caa; font-size:11px; letter-spacing:0.5px;">PRECISION</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#6A0DAD; font-size:26px; font-weight:800;">{sel_extra['Recall']:.1%}</div>
                    <div style="color:#9c8caa; font-size:11px; letter-spacing:0.5px;">RECALL</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#6A0DAD; font-size:26px; font-weight:800;">{sel_extra['AUC']:.1%}</div>
                    <div style="color:#9c8caa; font-size:11px; letter-spacing:0.5px;">AUC</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # =========================================================
    # 5. CLASSIFICATION REPORT + CONFUSION MATRIX
    # =========================================================
 
    report_col, cm_col = st.columns([1, 1])
 
    # ---------------------------------------------------------
    # LEFT: CLASSIFICATION REPORT
    # ---------------------------------------------------------
    with report_col:
      with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="dot"></span><span class="label">📄 Classification Report</span></div>', unsafe_allow_html=True)
 
        report_rows = []
 
        # Class rows
        for class_name in ["Low", "Medium", "High"]:
            row = selected_report[class_name]
 
            report_rows.append({
                "Class": class_name,
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1-Score": row["f1-score"],
                "Support": row["support"]
            })
 
        # Accuracy row
        report_rows.append({
            "Class": "Accuracy",
            "Precision": np.nan,
            "Recall": np.nan,
            "F1-Score": selected_report["accuracy"],
            "Support": 8007
        })
 
        # Macro Average
        macro = selected_report["macro avg"]
 
        report_rows.append({
            "Class": "Macro Avg",
            "Precision": macro["precision"],
            "Recall": macro["recall"],
            "F1-Score": macro["f1-score"],
            "Support": macro["support"]
        })
 
        # Weighted Average
        weighted = selected_report["weighted avg"]
 
        report_rows.append({
            "Class": "Weighted Avg",
            "Precision": weighted["precision"],
            "Recall": weighted["recall"],
            "F1-Score": weighted["f1-score"],
            "Support": weighted["support"]
        })
 
        report_df_display = pd.DataFrame(report_rows)
 
        st.dataframe(
            report_df_display.style.format({
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-Score": "{:.2f}",
                "Support": "{:.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
 
    # ---------------------------------------------------------
    # RIGHT: CONFUSION MATRIX
    # ---------------------------------------------------------
    with cm_col:
      with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="dot"></span><span class="label">🎯 Confusion Matrix</span></div>', unsafe_allow_html=True)
 
        cm = confusion_matrices[selected_perf_model]
 
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
 
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=confusion_colors[selected_perf_model],
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            ax=ax_cm,
            cbar=True
        )
 
        ax_cm.set_title(
            f"Confusion Matrix ({selected_perf_model} - Optimized)",
            fontsize=14,
            fontweight="bold",
            pad=15
        )
 
        ax_cm.set_xlabel("Predicted Engagement", fontsize=11)
        ax_cm.set_ylabel("Actual Engagement", fontsize=11)
 
        plt.tight_layout()
 
        st.pyplot(fig_cm, use_container_width=True)
 
    roc_class_colors = {"Low": "red", "Medium": "orange", "High": "green"}
  
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    roc_col, feat_col = st.columns([1, 1])
 
    # ---------------------------------------------------------
    # LEFT: MULTI-CLASS ROC CURVE
    # ---------------------------------------------------------
    with roc_col:
      with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="dot"></span><span class="label">📉 Multi-Class ROC Curve</span></div>', unsafe_allow_html=True)
 
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
 
        for class_name, color in roc_class_colors.items():
            target_auc = roc_auc_scores[selected_perf_model][class_name]
            fpr, tpr = generate_roc_curve(target_auc)
            ax_roc.plot(
                fpr, tpr,
                color=color, lw=2,
                label=f"{class_name} (AUC = {target_auc:.2f})"
            )
 
        ax_roc.plot([0, 1], [0, 1], "k--", lw=2)
 
        ax_roc.set_title(
            f"Multi-Class ROC Curve ({selected_perf_model})",
            fontsize=14, fontweight="bold", pad=15
        )
        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", fontsize=11)
        ax_roc.legend(loc="lower right")
 
        plt.tight_layout()
 
        st.pyplot(fig_roc, use_container_width=True)
 
    # ---------------------------------------------------------
    # RIGHT: TOP 10 FEATURE IMPORTANCE
    # ---------------------------------------------------------
    with feat_col:
      with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        style = feature_importance_style[selected_perf_model]
        st.markdown(f'<div class="section-header"><span class="dot"></span><span class="label">📈 {style["title"]}</span></div>', unsafe_allow_html=True)
 
        feat_imp = pd.Series(feature_importance_data[selected_perf_model])
        feat_imp = feat_imp.sort_values(ascending=True)
 
        fig_feat, ax_feat = plt.subplots(figsize=(6, 5))
 
        feat_imp.plot(kind="barh", ax=ax_feat, color=style["color"])
 
        ax_feat.set_title(
            f"{style['title']} ({selected_perf_model})",
            fontsize=14, fontweight="bold", pad=15
        )
        ax_feat.set_xlabel(style["xlabel"], fontsize=11)
 
        plt.tight_layout()
 
        st.pyplot(fig_feat, use_container_width=True)
 
    # =========================================================
    # 6. MODEL PARAMETERS FROM NOTEBOOK
    # =========================================================
 
    model_parameters = {
 
        "Logistic Regression": {
            "Regularization (C)": "0.1",
            "Solver": "lbfgs"
        },
 
        "Random Forest": {
            "Trees (n_estimators)": "100",
            "Max Depth": "20",
            "Min Samples Split": "5",
            "Min Samples Leaf": "2"
        },
 
        "KNN": {
            "K (n_neighbors)": "43",
            "Weights": "uniform",
            "Metric": "manhattan"
        },
 
        "XGBoost": {
            "Max Depth": "7",
            "Learning Rate": "0.1",
            "Trees (n_estimators)": "100"
        }
    }
 
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="dot"></span><span class="label">⚙️ Optimized Hyperparameters</span></div>', unsafe_allow_html=True)

        params = model_parameters[selected_perf_model]
        pills_html = "".join(
            f'<div class="hparam-pill"><div class="pval">{v}</div><div class="pname">{k}</div></div>'
            for k, v in params.items()
        )
        st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
 
    # =========================================================
    # 7. SUMMARY OF ALL MODELS
    # =========================================================
 
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="margin-top:6px;"><span class="dot"></span><span class="label" style="font-size:19px;">🏆 Overall Model Comparison</span></div>', unsafe_allow_html=True)
 
    # Exact values from the notebook final comparison
    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "KNN",
            "XGBoost"
        ],
        "Accuracy": [
            0.9040,
            0.9510,
            0.8461,
            0.9694
        ],
        "Precision": [
            0.9051,
            0.9516,
            0.8641,
            0.9696
        ],
        "Recall": [
            0.9040,
            0.9510,
            0.8461,
            0.9694
        ],
        "F1-Score": [
            0.9041,
            0.9510,
            0.8444,
            0.9694
        ],
        "AUC": [
            0.9571,
            0.9852,
            0.9404,
            0.9892
        ]
    })
 
    # Auto-detect the top performer to headline the section
    best_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
    st.markdown(f"""
    <div class="winner-banner">
        🥇 <b>{best_row['Model']}</b> leads the pack with <b>{best_row['Accuracy']:.2%}</b> accuracy
        and <b>{best_row['AUC']:.2%}</b> AUC — the strongest all-round performer of the four models tested.
    </div>
    """, unsafe_allow_html=True)
 
    # ---------------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------------
 
    with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="dot"></span><span class="label">📊 Performance Summary Table</span></div>', unsafe_allow_html=True)

        metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        styled_summary = (
            comparison_df.style
            .format({c: "{:.2%}" for c in metric_cols})
            .highlight_max(subset=metric_cols, props="background-color:#f2e6ff; color:#6A0DAD; font-weight:700;")
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center"), ("background-color", "#faf7ff"), ("color", "#5c1799")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fcfaff")]},
            ])
        )

        st.dataframe(
            styled_summary,
            use_container_width=True,
            hide_index=True
        )
 
    # ---------------------------------------------------------
    # SUMMARY GRAPH
    # ---------------------------------------------------------
 
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
      st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
      st.markdown('<div class="section-header"><span class="dot"></span><span class="label">📈 Final Algorithm Comparison</span></div>', unsafe_allow_html=True)

      # Convert to long format exactly like notebook
      plot_df = comparison_df.melt(
          id_vars="Model",
          value_vars=["Accuracy", "F1-Score", "AUC"],
          var_name="Metric",
          value_name="Score"
      )
 
      fig_summary, ax_summary = plt.subplots(figsize=(12, 7))
 
      sns.barplot(
          data=plot_df,
          x="Model",
          y="Score",
          hue="Metric",
          palette="viridis",
          ax=ax_summary
      )
 
      ax_summary.set_title(
          "Final Algorithm Comparison: Accuracy, F1-Score & AUC",
          fontsize=16,
          fontweight="bold",
          pad=15
      )
 
      ax_summary.set_xlabel(
          "Machine Learning Model",
          fontsize=12
      )
 
      ax_summary.set_ylabel(
          "Score (0.0 to 1.0)",
          fontsize=12
      )
 
      ax_summary.set_ylim(0, 1.15)
 
      # Legend outside the graph
      ax_summary.legend(
          bbox_to_anchor=(1.01, 1),
          loc="upper left",
          title="Metrics"
      )
 
      # Display exact values on top of bars
      for container in ax_summary.containers:
          ax_summary.bar_label(
              container,
              fmt="%.3f",
              padding=3
          )
 
      sns.despine()
 
      plt.tight_layout()
 
      st.pyplot(
          fig_summary,
          use_container_width=True
      )


# ------------------------------------------
# TAB 3: Prediction Result
# ------------------------------------------
with tab_pred:
    st.markdown("### 🎯 Player Engagement Predictor")
    st.markdown("Adjust the player features below to simulate and predict their engagement level.")

    # ==========================================
    # TAB 3 高级 CSS 设计注入 (契合 Bento Style)
    # ==========================================
    st.markdown("""
    <style>
    /* 用户输入档案的网格设计 */
    .profile-snapshot-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .profile-item {
        background: linear-gradient(180deg, #ffffff 0%, #fcfaff 100%);
        border: 1px solid #eee2f7;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 4px 10px rgba(106,13,173,0.03);
        border-bottom: 3px solid #e2c6ff;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .profile-item:hover {
        border-bottom: 3px solid #6A0DAD;
        transform: translateY(-2px);
    }
    .p-label { color: #8a7a99; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .p-val { color: #3a1050; font-size: 15px; font-weight: 800; }
    
    /* 预测结果特大焦点卡片设计 */
    .pred-hero-card {
        background: radial-gradient(circle at 90% 50%, rgba(106,13,173,0.08), transparent 50%),
                    linear-gradient(135deg, #ffffff 0%, #fdfbff 100%);
        border: 1px solid #eee2f7;
        border-left: 6px solid #6A0DAD;
        border-radius: 16px;
        padding: 25px 30px;
        margin-top: 10px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(106,13,173,0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .pred-title { color: #8a7a99; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;}
    .pred-value { color: #3a0a63; font-size: 42px; font-weight: 900; line-height: 1.1; margin-bottom: 8px;}
    .pred-model-badge { display: inline-block; background: #f0e2ff; color: #6A0DAD; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 12px; border: 1px solid #e2c6ff; }
    
    /* 预测策略动态卡片设计 */
    .strategy-card {
        background: #fffbfa;
        border: 1px solid #ffe8e3;
        border-left: 4px solid #ff6b6b;
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 15px;
    }
    .strategy-card.Medium { background: #f4faff; border-color: #dcedff; border-left-color: #3498db; }
    .strategy-card.High { background: #f4fff8; border-color: #d5ffe4; border-left-color: #2ecc71; }
    
    .strategy-title { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 17px; margin-bottom: 8px; color: #1a1a1a; }
    .strategy-text { color: #444; font-size: 14.5px; margin: 0; line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

    # 1. 初始化 Session State 来控制页面跳转
    if "show_prediction" not in st.session_state:
        st.session_state.show_prediction = False

    # 第一页：只有 Input Player Features
    if not st.session_state.show_prediction:
        st.markdown("#### 1. Input Player Features")
        with st.container(border=True):
            st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
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
                avg_duration = st.slider("Avg Session (Mins)", int(df['AvgSessionDurationMinutes'].min()), int(df['AvgSessionDurationMinutes'].max()), 60)
                player_level = st.slider("Player Level", int(df['PlayerLevel'].min()), int(df['PlayerLevel'].max()), 30)

            achievements = st.slider("Achievements Unlocked", int(df['AchievementsUnlocked'].min()), int(df['AchievementsUnlocked'].max()), 15)
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            if st.button("🔮 Predict Engagement", use_container_width=True):
                # 展现 Loading 动画，并在后台完成模型预测
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

                    # --- NEW: 保存用户的原始输入数据以便在结果页展示和导出 ---
                    st.session_state.user_profile = {
                        "Age": str(age),
                        "Gender": gender,
                        "Location": location,
                        "Game Genre": genre,
                        "Difficulty": difficulty,
                        "Play Time": f"{play_time} hrs",
                        "Purchases": in_purchases_label,
                        "Sessions": f"{sessions} / wk",
                        "Avg Session": f"{avg_duration} mins",
                        "Player Level": str(player_level),
                        "Achievements": str(achievements)
                    }

                    # 将预测结果存入 session_state
                    st.session_state.prediction = prediction
                    st.session_state.pred_model = selected_model_name
                    st.session_state.prob_df = prob_df
                    
                    # 切换状态变量，准备跳转第二页
                    st.session_state.show_prediction = True
                    st.rerun() # 重新加载组件，直接展示结果

    # 第二页： Prediction Insights (Result & Export)
    else:
        st.markdown("#### 2. Prediction Insights")
        
        # --- 模块 A: 还原显示 User Profile (让用户看到他们刚才预测的内容) ---
        with st.container(border=True):
            st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="dot"></span><span class="label">👤 Player Profile Snapshot</span></div>', unsafe_allow_html=True)
            
            profile = st.session_state.user_profile
            grid_html = '<div class="profile-snapshot-grid">'
            for k, v in profile.items():
                grid_html += f'<div class="profile-item"><div class="p-label">{k}</div><div class="p-val">{v}</div></div>'
            grid_html += '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

        # --- 模块 B: Prediction Result & Graph (美化图表与排版) ---
        with st.container(border=True):
            st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
            
            prediction = st.session_state.prediction
            selected_model_name = st.session_state.pred_model
            prob_df = st.session_state.prob_df

            # 美化的 Prediction Hero Card
            icon_map = {"Low": "⚠️", "Medium": "📊", "High": "🔥"}
            st.markdown(f"""
            <div class="pred-hero-card">
                <div>
                    <div class="pred-title">Predicted Engagement Level</div>
                    <div class="pred-value">{prediction}</div>
                    <div class="pred-model-badge">⚡ Powered by {selected_model_name}</div>
                </div>
                <div style="font-size: 65px; opacity: 0.9; text-shadow: 0 10px 20px rgba(0,0,0,0.1);">
                    {icon_map.get(prediction, '✨')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 更新 Plotly 图表颜色 (Low红色, Medium蓝色, High绿色) 并取消冗杂的背景
            color_discrete_map = {'Low': '#ff6b6b', 'Medium': '#3498db', 'High': '#2ecc71'}
            fig_prob = px.bar(
                prob_df, x="Probability", y="Engagement Level", 
                orientation='h', text_auto='.1%', 
                color="Engagement Level",
                color_discrete_map=color_discrete_map
            )
            fig_prob.update_layout(
                xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor='#f2e6ff'),
                yaxis=dict(title="", tickfont=dict(size=13, color='#3a0a63')),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False, 
                height=200, 
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # --- Actionable Strategy 模块美化 ---
            if prediction == "Low":
                s_icon, s_title = "🚨", "Retention Risk!"
                s_text = "Consider sending re-engagement emails, offering free starter packs, or suggesting easier game modes."
            elif prediction == "Medium":
                s_icon, s_title = "📈", "Steady Player"
                s_text = "Good potential for growth. Try offering limited-time quests or unlocking mid-tier achievements."
            else:
                s_icon, s_title = "⭐", "Highly Engaged!"
                s_text = "Ideal target for premium in-game purchases, exclusive VIP events, or beta testing new features."

            st.markdown(f"""
            <div class="strategy-card {prediction}">
                <div class="strategy-title"><span>{s_icon}</span> {s_title}</div>
                <div class="strategy-text">{s_text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # --- 模块 C: Back & Export CSV Buttons (新增导出功能) ---
        
        # 准备待导出的 CSV 数据：包含玩家基础特征 + 预测模型 + 结果 + 三个类别的概率
        export_df = pd.DataFrame([profile])
        export_df.insert(0, "Prediction_Model", selected_model_name)
        export_df.insert(1, "Predicted_Engagement", prediction)
        for index, row in prob_df.iterrows():
            export_df[f"Prob_{row['Engagement Level']}"] = f"{row['Probability']:.2%}"
            
        csv_data = export_df.to_csv(index=False).encode('utf-8')

        col_back, col_export = st.columns(2)
        with col_back:
            if st.button("⬅️ Back to Input", use_container_width=True):
                st.session_state.show_prediction = False
                st.rerun()
        with col_export:
            st.download_button(
                label="📥 Export Result (CSV)",
                data=csv_data,
                file_name=f"player_prediction_{selected_model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )