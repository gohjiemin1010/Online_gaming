import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
import time  # Imported for the loading spinner delay
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
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎮 Online Gaming Behavior Analysis & Prediction")

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
selected_model_name = st.sidebar.selectbox("Select Prediction Model", list(models_dict.keys()), index=1)

st.sidebar.markdown("**Player Features**")
age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 25)
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
location = st.sidebar.selectbox("Location", df['Location'].unique())
genre = st.sidebar.selectbox("Game Genre", df['GameGenre'].unique())
play_time = st.sidebar.number_input("Play Time (Hours)", 0.0, 24.0, 10.0)

# Changed 0/1 to No/Yes for better user experience
in_purchases_label = st.sidebar.selectbox("In-Game Purchases", ["No", "Yes"])
in_purchases = 1 if in_purchases_label == "Yes" else 0

difficulty = st.sidebar.selectbox("Game Difficulty", df['GameDifficulty'].unique())
sessions = st.sidebar.slider("Sessions Per Week", int(df['SessionsPerWeek'].min()), int(df['SessionsPerWeek'].max()), 5)
avg_duration = st.sidebar.slider("Avg Session Duration (Mins)", int(df['AvgSessionDurationMinutes'].min()), int(df['AvgSessionDurationMinutes'].max()), 60)
player_level = st.sidebar.slider("Player Level", int(df['PlayerLevel'].min()), int(df['PlayerLevel'].max()), 30)
achievements = st.sidebar.slider("Achievements Unlocked", int(df['AchievementsUnlocked'].min()), int(df['AchievementsUnlocked'].max()), 15)

predict_btn = st.sidebar.button("Predict Engagement", use_container_width=True)

# Add Loading Spinner and Success Prompt in the Sidebar
if predict_btn:
    with st.sidebar:
        with st.spinner("Analyzing player profile..."):
            time.sleep(1.2) # Small delay for visual loading effect
        st.success("Analysis complete! 👉 Please view the 'Prediction Result' tab on the right.")

# ==========================================
# 5. Main Content: Tabs Layout
# ==========================================
tab_eda, tab_perf, tab_pred = st.tabs([
    "Data Exploration", "Model Performance", "Prediction Result"
])

# ------------------------------------------
# TAB 1: Data Exploration
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
    
    # --- TABLE ON TOP (Data Summary) ---
    st.markdown("#### Statistical Summaries")
    summary_choice = st.radio("Select Summary View:", ["Numerical Summary", "Categorical Summary", "Dataset Preview"], horizontal=True)
    
    if summary_choice == "Dataset Preview":
        st.write("Use the +/- buttons or type a number to view more rows.")
        row_count = st.number_input("Number of rows to display:", min_value=5, max_value=len(df), value=100, step=10)
        st.dataframe(df.head(row_count), use_container_width=True)

    elif summary_choice == "Numerical Summary":
        num_desc = df.describe().T
        num_desc['range'] = num_desc['max'] - num_desc['min']
        num_desc['cv'] = (num_desc['std'] / num_desc['mean'] * 100).round(1)
        display_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'cv']
        st.dataframe(num_desc[display_cols].style.format("{:.2f}"), use_container_width=True)
        
    elif summary_choice == "Categorical Summary":
        cat_cols = df.select_dtypes(include=['object']).columns
        table_cols = st.columns(len(cat_cols))
        for i, col in enumerate(cat_cols):
            with table_cols[i]:
                st.markdown(f"**{col}**")
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, 'Count']
                st.dataframe(vc, hide_index=True, use_container_width=True)

    st.markdown("---")

    # --- GRAPH BELOW (Exploratory Visualizations) ---
    st.markdown("#### Exploratory Visualizations")
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
    
    st.markdown("#### Interactive Feature vs Feature Explorer")
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

    st.markdown("#### Correlation Heatmap")
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

    with report_col:

        st.markdown("#### Classification Report")

        report_rows = []

        for class_name in ["Low", "Medium", "High"]:
            row = selected_report[class_name]

            report_rows.append({
                "Class": class_name,
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1-Score": row["f1-score"],
                "Support": row["support"]
            })

        report_rows.append({
            "Class": "Accuracy",
            "Precision": np.nan,
            "Recall": np.nan,
            "F1-Score": selected_report["accuracy"],
            "Support": 8007
        })

        macro = selected_report["macro avg"]

        report_rows.append({
            "Class": "Macro Avg",
            "Precision": macro["precision"],
            "Recall": macro["recall"],
            "F1-Score": macro["f1-score"],
            "Support": macro["support"]
        })

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

    st.markdown("#### Final Algorithm Comparison")

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

    ax_summary.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        title="Metrics"
    )

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
        # 1. 准备输入数据
        input_data = pd.DataFrame([[age, gender, location, genre, play_time, in_purchases, 
                                    difficulty, sessions, avg_duration, player_level, achievements]], 
                                  columns=feature_cols)
        
        for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
            input_data[col] = le_dict[col].transform(input_data[col])
            
        input_scaled = scaler.transform(input_data)
        model = models_dict[selected_model_name]
        
        # 2. 获取预测类别和预测概率
        pred_encoded = model.predict(input_scaled)[0]
        prediction = le_dict['EngagementLevel'].inverse_transform([pred_encoded])[0]
        
        probabilities = model.predict_proba(input_scaled)[0]
        classes = le_dict['EngagementLevel'].inverse_transform(model.classes_)
        prob_df = pd.DataFrame({'Engagement Level': classes, 'Probability': probabilities})
        
        # 3. 布局设计：左边放预测结果和建议，右边放概率图
        col_res, col_prob = st.columns([1, 1.2])
        
        with col_res:
            st.metric(label=f"Predicted Engagement Level ({selected_model_name})", value=prediction)
            
            # 增加业务洞察/建议 (Business Recommendation)
            st.markdown("#### 💡 Actionable Insight")
            if prediction == "Low":
                st.warning("**Retention Risk!** Consider sending re-engagement emails, offering free starter packs, or suggesting easier game modes.")
            elif prediction == "Medium":
                st.info("**Steady Player.** Good potential for growth. Try offering limited-time quests or unlocking mid-tier achievements to increase sessions.")
            else:
                st.success("**Highly Engaged!** Ideal target for premium in-game purchases, exclusive VIP events, or beta testing new features.")
                
        with col_prob:
            st.markdown("#### Model Confidence (Probabilities)")
            # 使用横向柱状图展示模型对三个类别的预测概率
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
        
        # 4. 优化玩家资料的展示方式 (使用 4列 x 3行 的 Metric 卡片排版)
        st.markdown("#### 👤 Player Profile Evaluated")
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