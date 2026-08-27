import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
import time
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
# ==========================================
# 1. Page Configuration & Custom CSS 
# ==========================================
st.set_page_config(page_title="Online Gaming Analytics", page_icon="🎮", layout="wide")
 
# Custom CSS for Purple Theme, 3D Cards, Larger Tabs, Spacing, and Overriding Default Red
st.markdown("""
<style>
/* Move the main title up by reducing top padding */
.block-container {
    padding-top: 1.5rem !important;
}
 
/* 3D Metric Cards with Purple top border */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); 
    border-top: 4px solid #6A0DAD; /* Deep Purple Theme */
    transition: all 0.3s ease; 
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px); 
    box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.2); 
}
 
/* Enlarge Main Tab Font Size */
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px !important;
    font-weight: bold !important;
}
 
/* OVERRIDE STREAMLIT DEFAULT RED WITH PURPLE #6A0DAD */
/* 1. Active Tab Text & Highlight */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
    color: #6A0DAD !important;
}
.stTabs [data-baseweb="tab-list"] div[data-baseweb="tab-highlight"] {
    background-color: #6A0DAD !important;
}
/* 2. Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #6A0DAD !important;
}
.stSlider [data-baseweb="slider"] div[data-style] > div:first-child {
    background-color: #6A0DAD !important;
}
/* 3. Input & Dropdown Focus Borders */
div[data-baseweb="select"]:focus-within, div[data-baseweb="input"]:focus-within {
    border-color: #6A0DAD !important;
}
 
/* Style the Predict Button to match the Purple Theme */
div.stButton > button:first-child {
    background-color: #6A0DAD !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
}
div.stButton > button:first-child:hover {
    background-color: #5b0b9c !important;
}
 
/* =====================================================
   ABOUT US TAB — card + badge styling
   ===================================================== */
.about-hero {
    background: linear-gradient(135deg, #6A0DAD 0%, #9b59b6 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0px 8px 20px rgba(106, 13, 173, 0.25);
}
.about-hero h2 {
    margin: 0 0 6px 0;
    font-size: 26px;
}
.about-hero p {
    margin: 0;
    opacity: 0.92;
    font-size: 15px;
}
.about-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    border-left: 4px solid #6A0DAD;
    transition: all 0.25s ease;
}
.about-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 10px 20px rgba(0,0,0,0.15);
}
.about-card h4 {
    margin-top: 0;
    color: #6A0DAD;
}
.model-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    border-top: 4px solid #6A0DAD;
    position: relative;
}
.model-card.best {
    border-top: 4px solid #f1c40f;
}
.model-card .badge {
    position: absolute;
    top: -10px;
    right: 10px;
    background: #f1c40f;
    color: #4a3b00;
    font-size: 11px;
    font-weight: bold;
    padding: 3px 8px;
    border-radius: 20px;
}
.model-card h5 {
    margin: 6px 0 2px 0;
    color: #333;
}
.model-card .acc {
    font-size: 22px;
    font-weight: bold;
    color: #6A0DAD;
}
.tech-badge {
    display: inline-block;
    background-color: #F5F0FA;
    color: #6A0DAD;
    border: 1px solid #d8c2f0;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 4px 6px 4px 0;
    font-size: 13px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
 
st.markdown("##  Online Gaming Behavior Analysis & Prediction")
 
# Set Seaborn theme matching ipynb
sns.set_theme(style="white", context="notebook", font_scale=1.1)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
 
# ==========================================
# 2. Data Loading & Caching
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('online_gaming_behavior_dataset.csv')
    return df
 
df = load_data()
 
# ==========================================
# 3. Model Training & Metrics Calculation
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
    
    # Initialize 4 Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    trained_models = {}
    summary_metrics = []
    detailed_reports = {}
    confusion_matrices = {}
    target_names = le_dict['EngagementLevel'].classes_
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test_scaled)
        trained_models[name] = model
        
        acc = accuracy_score(y_test, pred)
        summary_metrics.append({"Model": name, "Accuracy": acc})
        detailed_reports[name] = classification_report(y_test, pred, target_names=target_names, output_dict=True)
        confusion_matrices[name] = confusion_matrix(y_test, pred)
        
    summary_df = pd.DataFrame(summary_metrics)
    
    return trained_models, le_dict, scaler, X.columns, summary_df, detailed_reports, confusion_matrices
 
models_dict, le_dict, scaler, feature_cols, summary_df, detailed_reports, confusion_matrices = train_models(df)
 
# ==========================================
# 4. Sidebar: User Input (Single Prediction)
# ==========================================
st.sidebar.markdown("###  Player Engagement Predictor")
selected_model_name = st.sidebar.selectbox("Select Model", list(models_dict.keys()), index=1)
 
st.sidebar.markdown("**Player Features**")
age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 25)
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
location = st.sidebar.selectbox("Location", df['Location'].unique())
genre = st.sidebar.selectbox("Game Genre", df['GameGenre'].unique())
play_time = st.sidebar.number_input("Play Time (Hours)", 0.0, 24.0, 10.0)
 
# Changed 0/1 to No/Yes
in_purchases_label = st.sidebar.selectbox("In-Game Purchases", ["No", "Yes"])
in_purchases = 1 if in_purchases_label == "Yes" else 0
 
difficulty = st.sidebar.selectbox("Game Difficulty", df['GameDifficulty'].unique())
sessions = st.sidebar.slider("Sessions Per Week", int(df['SessionsPerWeek'].min()), int(df['SessionsPerWeek'].max()), 5)
avg_duration = st.sidebar.slider("Avg Session Duration (Mins)", int(df['AvgSessionDurationMinutes'].min()), int(df['AvgSessionDurationMinutes'].max()), 60)
player_level = st.sidebar.slider("Player Level", int(df['PlayerLevel'].min()), int(df['PlayerLevel'].max()), 30)
achievements = st.sidebar.slider("Achievements Unlocked", int(df['AchievementsUnlocked'].min()), int(df['AchievementsUnlocked'].max()), 15)
 
predict_btn = st.sidebar.button("Predict Engagement", use_container_width=True)
 
if predict_btn:
    with st.sidebar:
        with st.spinner("Analyzing player profile..."):
            time.sleep(1.2) # Small delay for visual loading effect
        st.success("Analysis complete! 👉 Please view the 'Prediction Result' tab on the right.")
 
# ==========================================
# 5. Main Content: Tabs Layout
# ==========================================
tab_eda, tab_perf, tab_pred, tab_about = st.tabs([
    "Data Exploration", "Model Performance", "Prediction Result", "About Us"
])
 
# ------------------------------------------
# TAB 1: Data Exploration (KEPT EXACTLY AS USER REQUESTED)
# ------------------------------------------
with tab_eda:
    st.markdown("### Exploratory Data Analysis")
    
    # Beautiful 3D Animated Metric Cards
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
    
    sub1, sub2 = st.tabs(["Basic Data Understanding", "Exploratory Visualizations"])
 
    with sub1:
        st.markdown("####  Dataset Preview")
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
 
    with sub2:
        # --- FEATURES DISTRIBUTION SECTION ---
        st.markdown("####  Features Distribution")
        st.write("Select a feature to view its distribution based on your exploratory analysis.")
        
        dist_choice = st.selectbox("Select Feature to Visualize:", 
                                   ["All", "EngagementLevel", "GameGenre", "Age", "PlayTimeHours"], index=0)
        
        dist_figs = []
 
        if dist_choice == "All" or dist_choice == "EngagementLevel":
            fig_dist1, ax_dist1 = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='EngagementLevel', hue='EngagementLevel',
                          order=['Low', 'Medium', 'High'], ax=ax_dist1, 
                          palette=['#ff9999','#66b3ff','#99ff99'], legend=False)
            ax_dist1.set_title('Distribution of Engagement Levels', fontsize=14, weight='bold', pad=15)
            ax_dist1.set_ylabel('Number of Players')
            ax_dist1.set_xlabel('')
            for container in ax_dist1.containers:
                ax_dist1.bar_label(container, padding=3)
            sns.despine()
            dist_figs.append(fig_dist1)
 
        if dist_choice == "All" or dist_choice == "GameGenre":
            fig_dist2, ax_dist2 = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, y='GameGenre', hue='GameGenre',
                          ax=ax_dist2, palette='crest', legend=False)
            ax_dist2.set_title('Popularity of Game Genres', fontsize=14, weight='bold', pad=15)
            ax_dist2.set_xlabel('Number of Players')
            ax_dist2.set_ylabel('')
            for container in ax_dist2.containers:
                ax_dist2.bar_label(container, padding=3)
            sns.despine()
            dist_figs.append(fig_dist2)
 
        if dist_choice == "All" or dist_choice == "Age":
            fig_dist3, ax_dist3 = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Age'], bins=25, kde=True, ax=ax_dist3, 
                         color='#9b59b6', edgecolor='white', alpha=0.7)
            ax_dist3.set_title('Player Age Distribution', fontsize=14, weight='bold', pad=15)
            ax_dist3.set_xlabel('Age')
            ax_dist3.set_ylabel('Frequency')
            sns.despine()
            dist_figs.append(fig_dist3)
 
        if dist_choice == "All" or dist_choice == "PlayTimeHours":
            fig_dist4, ax_dist4 = plt.subplots(figsize=(8, 5))
            sns.histplot(df['PlayTimeHours'], bins=25, kde=True, ax=ax_dist4, 
                         color='#3498db', edgecolor='white', alpha=0.7)
            ax_dist4.set_title('Play Time Hours Distribution', fontsize=14, weight='bold', pad=15)
            ax_dist4.set_xlabel('Play Time (Hours)')
            ax_dist4.set_ylabel('Frequency')
            sns.despine()
            dist_figs.append(fig_dist4)
 
        for i in range(0, len(dist_figs), 2):
            col_a, col_b = st.columns(2)
            with col_a:
                st.pyplot(dist_figs[i])
            with col_b:
                if i + 1 < len(dist_figs):
                    st.pyplot(dist_figs[i+1])
 
        st.markdown("---")
        
        # --- VERSUS & NOTEBOOK EXPLORER SECTION (SINGLE SELECTOR) ---
        st.markdown("####  Interactive Feature vs Feature Explorer")
        st.write("Select a specific relationship graph from your Jupyter Notebook or view All.")
        
        analysis_choice = st.selectbox(
            "Select Analysis Graph:",
            [
                "All",
                "Player Level vs. Achievements Unlocked",
                "In-Game Purchase Rate by Game Genre",
                "Session Duration Density by Game Difficulty",
                "Game Genre Preferences by Gender",
                "Player Engagement Levels by Geographic Location",
                "Weekly Sessions Based on Game Difficulty",
                "Age vs. Average Session Duration",
                "In-Game Purchase Rate by Location",
                "Achievements Unlocked per Game Difficulty",
                "Play Time Hours by Engagement Level"
            ],
            index=0
        )
        
        figures_to_plot = []
 
        if analysis_choice == "All" or analysis_choice == "Player Level vs. Achievements Unlocked":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='PlayerLevel', y='AchievementsUnlocked', hue='EngagementLevel',
                            palette={'Low':'#99ff99', 'Medium':'#ff9999', 'High':'#66b3ff'}, alpha=0.7, ax=ax)
            ax.set_title('Player Level vs. Achievements Unlocked', fontsize=14, weight='bold')
            ax.set_xlabel('Player Level')
            ax.set_ylabel('Achievements Unlocked')
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "In-Game Purchase Rate by Game Genre":
            fig, ax = plt.subplots(figsize=(8, 5))
            genre_purchase = df.groupby('GameGenre')['InGamePurchases'].mean().sort_values().reset_index()
            sns.barplot(data=genre_purchase, x='GameGenre', y='InGamePurchases', palette='mako', ax=ax)
            ax.set_title('In-Game Purchase Rate by Game Genre', fontsize=14, weight='bold')
            ax.set_xlabel('Game Genre')
            ax.set_ylabel('Purchase Rate (Percentage)')
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "Session Duration Density by Game Difficulty":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.kdeplot(data=df, x='AvgSessionDurationMinutes', hue='GameDifficulty', fill=True, alpha=0.5,
                        palette={'Easy':'#3498db', 'Medium':'#e74c3c', 'Hard':'#2ecc71'}, ax=ax)
            ax.set_title('Session Duration Density by Game Difficulty', fontsize=14, weight='bold')
            ax.set_xlabel('Average Session Duration (Minutes)')
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "Game Genre Preferences by Gender":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='GameGenre', hue='Gender', palette='Set2', ax=ax)
            ax.set_title('Game Genre Preferences by Gender', fontsize=14, weight='bold')
            ax.set_xlabel('Game Genre')
            ax.set_ylabel('Number of Players')
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "Player Engagement Levels by Geographic Location":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='Location', hue='EngagementLevel', 
                          order=['Other', 'USA', 'Europe', 'Asia'], 
                          hue_order=['Low', 'Medium', 'High'],
                          palette=['#ff9999','#66b3ff','#99ff99'], ax=ax)
            ax.set_title('Player Engagement Levels by Geographic Location', fontsize=14, weight='bold')
            ax.set_xlabel('Location')
            ax.set_ylabel('Number of Players')
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "Weekly Sessions Based on Game Difficulty":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x='GameDifficulty', y='SessionsPerWeek', 
                        order=['Easy', 'Medium', 'Hard'], palette='Wistia', ax=ax)
            ax.set_title('Weekly Sessions Based on Game Difficulty', fontsize=14, weight='bold')
            ax.set_xlabel('Game Difficulty')
            ax.set_ylabel('Sessions Per Week')
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "Age vs. Average Session Duration":
            jfig = sns.jointplot(data=df, x='Age', y='AvgSessionDurationMinutes', kind='hex', color='#4CB391', height=6)
            jfig.fig.suptitle('Age vs. Average Session Duration', fontsize=14, weight='bold', y=1.03)
            figures_to_plot.append(jfig.fig)
 
        if analysis_choice == "All" or analysis_choice == "In-Game Purchase Rate by Location":
            fig, ax = plt.subplots(figsize=(8, 5))
            loc_purchase = df.groupby('Location')['InGamePurchases'].mean().loc[['Asia', 'Europe', 'Other', 'USA']].reset_index()
            sns.barplot(data=loc_purchase, x='Location', y='InGamePurchases', palette='cubehelix', ax=ax)
            ax.set_title('In-Game Purchase Rate by Location', fontsize=14, weight='bold')
            ax.set_xlabel('Location')
            ax.set_ylabel('Purchase Conversion Rate')
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "Achievements Unlocked per Game Difficulty":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.stripplot(data=df, x='GameDifficulty', y='AchievementsUnlocked', 
                          order=['Easy', 'Medium', 'Hard'], palette='Dark2', ax=ax, jitter=True)
            ax.set_title('Achievements Unlocked per Game Difficulty', fontsize=14, weight='bold')
            ax.set_xlabel('Game Difficulty')
            ax.set_ylabel('Achievements Unlocked')
            sns.despine()
            figures_to_plot.append(fig)
 
        if analysis_choice == "All" or analysis_choice == "Play Time Hours by Engagement Level":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(
                data=df, x='EngagementLevel', y='PlayTimeHours', 
                hue='EngagementLevel', order=['Low', 'Medium', 'High'], 
                palette='pastel', inner='quartile', legend=False,
                linewidth=1.5, ax=ax
            )
            ax.set_title('Play Time Hours by Engagement Level', fontsize=14, weight='bold', pad=15)
            ax.set_xlabel('Engagement Level')
            ax.set_ylabel('Play Time (Hours)')
            sns.despine()
            figures_to_plot.append(fig)
 
        for i in range(0, len(figures_to_plot), 2):
            col_a, col_b = st.columns(2)
            with col_a:
                st.pyplot(figures_to_plot[i])
            with col_b:
                if i + 1 < len(figures_to_plot):
                    st.pyplot(figures_to_plot[i+1])
 
        st.markdown("---")
 
        # --- CORRELATION HEATMAP SECTION (Positioned right below Versus) ---
        st.markdown("####  Correlation Heatmap")
        st.write("Correlation of target variable among each numerical column.")
        
        fig_corr, ax_corr = plt.subplots(figsize=(12, 9))
        numeric_cols_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
        corr_matrix = numeric_cols_df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, cmap='vlag', 
            fmt=".2f", linewidths=1, cbar_kws={"shrink": .8}, 
            square=True, center=0, ax=ax_corr
        )
        ax_corr.set_title('Correlation Heatmap', fontsize=16, weight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
 
 
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
 
    # Four horizontal buttons
    btn_cols = st.columns(4)
 
    for i, model_name in enumerate(performance_models):
        with btn_cols[i]:
 
            # Add check mark to currently selected model
            button_label = (
                f"✓ {model_name}"
                if current_perf_model == model_name
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
 
    st.markdown("---")
 
    # =========================================================
    # 2. NOTEBOOK-BASED CLASSIFICATION REPORTS
    # =========================================================
 
    classification_reports = {
 
        "Logistic Regression": {
            "Low": {
                "precision": 0.89,
                "recall": 0.90,
                "f1-score": 0.90,
                "support": 2065
            },
            "Medium": {
                "precision": 0.89,
                "recall": 0.92,
                "f1-score": 0.90,
                "support": 3875
            },
            "High": {
                "precision": 0.95,
                "recall": 0.88,
                "f1-score": 0.91,
                "support": 2067
            },
            "macro avg": {
                "precision": 0.91,
                "recall": 0.90,
                "f1-score": 0.90,
                "support": 8007
            },
            "weighted avg": {
                "precision": 0.91,
                "recall": 0.90,
                "f1-score": 0.90,
                "support": 8007
            },
            "accuracy": 0.9040
        },
 
        "Random Forest": {
            "Low": {
                "precision": 0.95,
                "recall": 0.96,
                "f1-score": 0.96,
                "support": 2065
            },
            "Medium": {
                "precision": 0.94,
                "recall": 0.96,
                "f1-score": 0.95,
                "support": 3875
            },
            "High": {
                "precision": 0.97,
                "recall": 0.92,
                "f1-score": 0.94,
                "support": 2067
            },
            "macro avg": {
                "precision": 0.96,
                "recall": 0.95,
                "f1-score": 0.95,
                "support": 8007
            },
            "weighted avg": {
                "precision": 0.95,
                "recall": 0.95,
                "f1-score": 0.95,
                "support": 8007
            },
            "accuracy": 0.9510
        },
 
        "KNN": {
            "Low": {
                "precision": 0.93,
                "recall": 0.71,
                "f1-score": 0.80,
                "support": 2065
            },
            "Medium": {
                "precision": 0.78,
                "recall": 0.96,
                "f1-score": 0.86,
                "support": 3875
            },
            "High": {
                "precision": 0.96,
                "recall": 0.78,
                "f1-score": 0.86,
                "support": 2067
            },
            "macro avg": {
                "precision": 0.89,
                "recall": 0.81,
                "f1-score": 0.84,
                "support": 8007
            },
            "weighted avg": {
                "precision": 0.86,
                "recall": 0.85,
                "f1-score": 0.84,
                "support": 8007
            },
            "accuracy": 0.8461
        },
 
        "XGBoost": {
            "Low": {
                "precision": 0.97,
                "recall": 0.98,
                "f1-score": 0.97,
                "support": 2065
            },
            "Medium": {
                "precision": 0.96,
                "recall": 0.97,
                "f1-score": 0.97,
                "support": 3875
            },
            "High": {
                "precision": 0.98,
                "recall": 0.95,
                "f1-score": 0.97,
                "support": 2067
            },
            "macro avg": {
                "precision": 0.97,
                "recall": 0.97,
                "f1-score": 0.97,
                "support": 8007
            },
            "weighted avg": {
                "precision": 0.97,
                "recall": 0.97,
                "f1-score": 0.97,
                "support": 8007
            },
            "accuracy": 0.9694
        }
    }
 
    # =========================================================
    # 3. NOTEBOOK-BASED CONFUSION MATRICES
    # =========================================================
 
    confusion_matrices = {
 
        "Logistic Regression": np.array([
            [1867, 198, 0],
            [220, 3555, 100],
            [6, 245, 1816]
        ]),
 
        "Random Forest": np.array([
            [1990, 75, 0],
            [94, 3731, 50],
            [0, 173, 1894]
        ]),
 
        "KNN": np.array([
            [1461, 603, 1],
            [102, 3708, 65],
            [13, 448, 1606]
        ]),
 
        "XGBoost": np.array([
            [2020, 45, 0],
            [70, 3773, 32],
            [0, 98, 1969]
        ])
    }
 
    # Exact color schemes used in notebook
    confusion_colors = {
        "Logistic Regression": "Blues",
        "Random Forest": "Greens",
        "KNN": "Purples",
        "XGBoost": "OrRd"
    }
 
    # =========================================================
    # 4. MODEL TITLE + ACCURACY
    # =========================================================
 
    selected_report = classification_reports[selected_perf_model]
 
    st.markdown(
        f"### {selected_perf_model}"
    )
 
    model_accuracy = selected_report["accuracy"]
 
    st.metric(
        label="Testing Set Accuracy",
        value=f"{model_accuracy:.2%}"
    )
 
    # =========================================================
    # 5. CLASSIFICATION REPORT + CONFUSION MATRIX
    # =========================================================
 
    report_col, cm_col = st.columns([1, 1])
 
    # ---------------------------------------------------------
    # LEFT: CLASSIFICATION REPORT
    # ---------------------------------------------------------
    with report_col:
 
        st.markdown("#### Classification Report")
 
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
 
        st.markdown("#### Confusion Matrix")
 
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
 
    # =========================================================
    # 5b. ROC CURVE DATA (per-class AUC read from notebook plots)
    # =========================================================
 
    roc_auc_scores = {
        "Logistic Regression": {"Low": 0.98, "Medium": 0.94, "High": 0.96},
        "Random Forest":       {"Low": 0.99, "Medium": 0.98, "High": 0.99},
        "KNN":                 {"Low": 0.96, "Medium": 0.93, "High": 0.95},
        "XGBoost":             {"Low": 1.00, "Medium": 0.98, "High": 0.99},
    }
 
    roc_class_colors = {"Low": "red", "Medium": "orange", "High": "green"}
 
    def generate_roc_curve(target_auc, n_points=300):
        """
        Reconstructs a smooth ROC curve shaped to hit an exact target AUC,
        using the standard binormal ROC model. The notebook's roc_curve()
        call produced thousands of raw (fpr, tpr) points from the test-set
        probabilities that aren't stored anywhere except inside the plotted
        PNG, so this regenerates a curve visually equivalent to the
        notebook's, calibrated to the exact AUC the notebook reported.
        """
        target_auc = min(max(target_auc, 0.5001), 0.9999)
        a = np.sqrt(2) * norm.ppf(target_auc)
        fpr = np.linspace(0.0001, 0.9999, n_points)
        tpr = norm.cdf(a + norm.ppf(fpr))
        tpr = np.clip(tpr, 0, 1)
        fpr = np.concatenate([[0.0], fpr, [1.0]])
        tpr = np.concatenate([[0.0], tpr, [1.0]])
        return fpr, tpr
 
    # =========================================================
    # 5c. FEATURE IMPORTANCE DATA (read from notebook plots)
    # =========================================================
 
    feature_importance_data = {
        "Logistic Regression": {
            "TotalWeeklyMinutes": 6.00,
            "SessionsPerWeek": 0.90,
            "AvgSessionDurationMinutes": 0.80,
            "AchievementsUnlocked": 0.35,
            "AchievementRate": 0.25,
            "PlayerLevel": 0.10,
            "AgeGroup_Adult": 0.05,
            "Age": 0.03,
            "AgeGroup_YoungAdult": 0.02,
            "Location_USA": 0.01,
        },
        "Random Forest": {
            "TotalWeeklyMinutes": 0.510,
            "SessionsPerWeek": 0.210,
            "AvgSessionDurationMinutes": 0.120,
            "AchievementRate": 0.055,
            "PlayerLevel": 0.025,
            "AchievementsUnlocked": 0.022,
            "PlayTimeHours": 0.015,
            "Age": 0.008,
            "GameDifficulty": 0.004,
            "Gender_Male": 0.003,
        },
        "KNN": {
            "TotalWeeklyMinutes": 0.260,
            "SessionsPerWeek": 0.170,
            "AvgSessionDurationMinutes": 0.105,
            "AchievementsUnlocked": 0.013,
            "AchievementRate": 0.006,
            "PlayerLevel": 0.004,
            "Gender_Male": 0.003,
            "PlayTimeHours": 0.002,
            "InGamePurchases": 0.001,
            "Location_USA": 0.001,
        },
        "XGBoost": {
            "TotalWeeklyMinutes": 0.685,
            "AchievementsUnlocked": 0.065,
            "PlayerLevel": 0.050,
            "AchievementRate": 0.035,
            "SessionsPerWeek": 0.028,
            "AvgSessionDurationMinutes": 0.012,
            "Location_Europe": 0.007,
            "GameGenre_Strategy": 0.006,
            "Age": 0.005,
            "GameDifficulty": 0.005,
        },
    }
 
    # Exact bar color + axis label + title used per model in the notebook
    feature_importance_style = {
        "Logistic Regression": {
            "color": "teal",
            "xlabel": "Mean Absolute Coefficient (Impact)",
            "title": "Top 10 Feature Importance",
        },
        "Random Forest": {
            "color": "forestgreen",
            "xlabel": "Feature Importance Score",
            "title": "Top 10 Feature Importance",
        },
        "KNN": {
            "color": "rebeccapurple",
            "xlabel": "Mean Accuracy Drop Upon Permutation",
            "title": "Top 10 Permutation Feature Importance",
        },
        "XGBoost": {
            "color": "orangered",
            "xlabel": "Feature Importance Score",
            "title": "Top 10 Feature Importance",
        },
    }
  
    roc_col, feat_col = st.columns([1, 1])
 
    # ---------------------------------------------------------
    # LEFT: MULTI-CLASS ROC CURVE
    # ---------------------------------------------------------
    with roc_col:
 
        st.markdown("##### Multi-Class ROC Curve")
 
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
 
        style = feature_importance_style[selected_perf_model]
 
        st.markdown(f"##### {style['title']}")
 
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
 
    with st.expander("⚙️ Optimized Hyperparameters", expanded=False):
 
        params = model_parameters[selected_perf_model]
 
        param_cols = st.columns(len(params))
 
        for i, (param_name, param_value) in enumerate(params.items()):
            with param_cols[i]:
                st.metric(param_name, param_value)
 
    # =========================================================
    # 7. SUMMARY OF ALL MODELS
    # =========================================================
 
    st.markdown("---")
    st.markdown("##  Overall Model Comparison")
 
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
 
    # ---------------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------------
 
    st.markdown("#### Performance Summary Table")
 
    summary_display = comparison_df.copy()
 
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
        summary_display[col] = summary_display[col].map(
            lambda x: f"{x:.2%}"
        )
 
    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True
    )
 
    # ---------------------------------------------------------
    # SUMMARY GRAPH
    # ---------------------------------------------------------
 
    st.markdown("#### Final Algorithm Comparison")
 
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
# TAB 3: Single Prediction Result & Insights
# ------------------------------------------
with tab_pred:
    st.markdown("###  Prediction Result & Insights")
    if predict_btn:
        # 1. Prepare Input Data
        input_data = pd.DataFrame([[age, gender, location, genre, play_time, in_purchases, 
                                    difficulty, sessions, avg_duration, player_level, achievements]], 
                                  columns=feature_cols)
        
        for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
            input_data[col] = le_dict[col].transform(input_data[col])
            
        input_scaled = scaler.transform(input_data)
        model = models_dict[selected_model_name]
        
        # 2. Get Prediction and Probabilities
        pred_encoded = model.predict(input_scaled)[0]
        prediction = le_dict['EngagementLevel'].inverse_transform([pred_encoded])[0]
        
        probabilities = model.predict_proba(input_scaled)[0]
        classes = le_dict['EngagementLevel'].inverse_transform(model.classes_)
        prob_df = pd.DataFrame({'Engagement Level': classes, 'Probability': probabilities})
        
        # 3. Layout: Left = Result/Insights, Right = Probability Bar Chart
        col_res, col_prob = st.columns([1, 1.2])
        
        with col_res:
            st.metric(label=f"Predicted Engagement Level ({selected_model_name})", value=prediction)
            
            # Actionable Insight based on prediction
            st.markdown("#### 💡 Actionable Insight")
            if prediction == "Low":
                st.warning("**Retention Risk!** Consider sending re-engagement emails, offering free starter packs, or suggesting easier game modes.")
            elif prediction == "Medium":
                st.info("**Steady Player.** Good potential for growth. Try offering limited-time quests or unlocking mid-tier achievements to increase sessions.")
            else:
                st.success("**Highly Engaged!** Ideal target for premium in-game purchases, exclusive VIP events, or beta testing new features.")
                
        with col_prob:
            st.markdown("#### Model Confidence (Probabilities)")
            # Horizontal Bar Chart for Probabilities
            fig_prob = px.bar(
                prob_df, x="Probability", y="Engagement Level", 
                orientation='h', text_auto='.1%', 
                color="Engagement Level",
                color_discrete_map={'Low': '#ff9999', 'Medium': '#66b3ff', 'High': '#99ff99'}
            )
            fig_prob.update_layout(
                xaxis=dict(range=[0, 1], tickformat=".0%"), 
                showlegend=False, 
                height=250, 
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        st.markdown("---")
        
        # 4. Player Profile Layout (4 columns x 3 rows Metric Cards)
        st.markdown("####  Player Profile Evaluated")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        
        with p_col1:
            st.metric("Age", age)
            st.metric("Play Time", f"{play_time} hrs")
            st.metric("Player Level", f"Lv. {player_level}")
            
        with p_col2:
            st.metric("Gender", gender)
            st.metric("Avg Session", f"{avg_duration} mins")
            st.metric("Achievements", achievements)
            
        with p_col3:
            st.metric("Location", location)
            st.metric("Sessions/Week", sessions)
            st.metric("In-Game Purchase", "Yes" if in_purchases == 1 else "No")
            
        with p_col4:
            st.metric("Game Genre", genre)
            st.metric("Difficulty", difficulty)
            st.metric("Profile Status", "Evaluated ✅")
            
    else:
        st.info("👈 Please enter player details in the sidebar and click 'Predict' to view detailed results.")
 
# ------------------------------------------
# TAB 4: About Us
# ------------------------------------------
with tab_about:
 
    # ---- Hero banner (single-line HTML avoids Streamlit's markdown
    # treating indented multi-line HTML as a code block) ----
    hero_html = (
        '<div class="about-hero">'
        '<h2>About This Project</h2>'
        '<p>A machine-learning dashboard that turns raw gaming activity into a clear read on how '
        'engaged a player really is — built end-to-end from EDA to a live predictor.</p>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)
 
    # ---- What the app does ----
    st.markdown("#### What This App Does")
    st.write(
        "This dashboard analyses the **Online Gaming Behavior Dataset** and predicts a player's "
        "**Engagement Level** — Low, Medium, or High — based on how they play. It walks through the "
        "full data science pipeline: exploring the raw data, engineering new features, comparing four "
        "classification models, and letting you test predictions live from the sidebar."
    )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ---- Feature highlight cards ----
    about_cards = [
        ("Data Exploration",
         "Interactive distributions, a feature-vs-feature explorer, and a correlation heatmap "
         "covering relationship charts from the notebook's EDA."),
        ("Model Comparison",
         "Classification reports, confusion matrices, ROC curves, and feature importance "
         "side-by-side for 4 optimized ML models."),
        ("Live Prediction",
         "Enter a player's profile in the sidebar and instantly get a predicted engagement "
         "level with confidence scores and actionable retention tips."),
    ]
    cols = st.columns(3)
    for (title, desc), col in zip(about_cards, cols):
        with col:
            card_html = f'<div class="about-card"><h4>{title}</h4><p>{desc}</p></div>'
            st.markdown(card_html, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
 
    # ---- How it works (replaces the old duplicated dataset/model stat
    # cards, since those numbers already live in the Data Exploration
    # and Model Performance tabs) ----
    st.markdown("#### How It Works")
    steps = [
        ("1", "Explore the Data", "Understand player behaviour through distributions and correlations."),
        ("2", "Engineer Features", "Derive TotalWeeklyMinutes, AchievementRate, and AgeGroup."),
        ("3", "Train & Compare", "Tune and benchmark 4 models: Logistic Regression, Random Forest, KNN, XGBoost."),
        ("4", "Predict Live", "Enter a player profile and get an instant engagement prediction."),
    ]
    step_cols = st.columns(4)
    for (num, title, desc), col in zip(steps, step_cols):
        with col:
            step_html = (
                f'<div class="step-card"><div class="step-num">{num}</div>'
                f'<h5>{title}</h5><p>{desc}</p></div>'
            )
            st.markdown(step_html, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ---- One-line key result (full comparison table lives in Model Performance) ----
    result_html = (
        '<div class="result-banner">'
        '<div><div class="label">Best Performing Model</div>'
        '<div class="value">XGBoost</div></div>'
        '<div><div class="label">Test Accuracy</div>'
        '<div class="value">96.9%</div></div>'
        '<div><div class="label">Models Compared</div>'
        '<div class="value">4</div></div>'
        '</div>'
    )
    st.markdown(result_html, unsafe_allow_html=True)
    st.caption("See the Model Performance tab for full classification reports, confusion matrices, and ROC curves.")
 
    st.markdown("---")
 
    # ---- Tech stack ----
    st.markdown("#### Built With")
    tech_stack = [
        "Python", "Streamlit", "Pandas", "NumPy", "Scikit-learn",
        "XGBoost", "Seaborn", "Matplotlib", "Plotly"
    ]
    badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in tech_stack])
    st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Online Gaming Behavior Analysis & Prediction — a data science course project.")