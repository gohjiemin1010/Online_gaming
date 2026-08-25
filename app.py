import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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
    target_names = le_dict['EngagementLevel'].classes_
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test_scaled)
        trained_models[name] = model
        
        acc = accuracy_score(y_test, pred)
        summary_metrics.append({"Model": name, "Accuracy": acc})
        detailed_reports[name] = classification_report(y_test, pred, target_names=target_names, output_dict=True)
        
    summary_df = pd.DataFrame(summary_metrics)
    
    return trained_models, le_dict, scaler, X.columns, summary_df, detailed_reports

models_dict, le_dict, scaler, feature_cols, summary_df, detailed_reports = train_models(df)

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
in_purchases = st.sidebar.selectbox("In-Game Purchases", [0, 1])
difficulty = st.sidebar.selectbox("Game Difficulty", df['GameDifficulty'].unique())
sessions = st.sidebar.slider("Sessions Per Week", int(df['SessionsPerWeek'].min()), int(df['SessionsPerWeek'].max()), 5)
avg_duration = st.sidebar.slider("Avg Session Duration (Mins)", int(df['AvgSessionDurationMinutes'].min()), int(df['AvgSessionDurationMinutes'].max()), 60)
player_level = st.sidebar.slider("Player Level", int(df['PlayerLevel'].min()), int(df['PlayerLevel'].max()), 30)
achievements = st.sidebar.slider("Achievements Unlocked", int(df['AchievementsUnlocked'].min()), int(df['AchievementsUnlocked'].max()), 15)

predict_btn = st.sidebar.button("Predict Engagement", use_container_width=True)

# Quick Metrics in Sidebar
with st.sidebar.expander("📊 Quick Model Accuracy"):
    current_acc = summary_df[summary_df['Model'] == selected_model_name]['Accuracy'].values[0]
    st.markdown(f"**Model:** {selected_model_name}")
    st.markdown(f"**Accuracy:** {current_acc:.2%}")

# ==========================================
# 5. Main Content: Tabs Layout (Reordered)
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
    
    # Sub-tabs for detailed EDA (Cleaned up as requested)
    sub1, sub2, sub3 = st.tabs(["Basic Data Understanding", "Features & Versus Explorer", "Correlation Analysis"])

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
        
        # Dropdown restricted to exactly 4 features from the ipynb
        dist_choice = st.selectbox("Select Feature to Visualize:", 
                                   ["EngagementLevel", "GameGenre", "Age", "PlayTimeHours"])
        
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        
        # Plot styling strictly matches the provided ipynb output
        if dist_choice == "EngagementLevel":
            sns.countplot(data=df, x='EngagementLevel', hue='EngagementLevel',
                          order=['Low', 'Medium', 'High'], ax=ax_dist, 
                          palette=['#ff9999','#66b3ff','#99ff99'], legend=False)
            ax_dist.set_title('Distribution of Engagement Levels', fontsize=14, weight='bold', pad=15)
            ax_dist.set_ylabel('Number of Players')
            ax_dist.set_xlabel('')
            for container in ax_dist.containers:
                ax_dist.bar_label(container, padding=3)

        elif dist_choice == "GameGenre":
            sns.countplot(data=df, y='GameGenre', hue='GameGenre',
                          ax=ax_dist, palette='crest', legend=False)
            ax_dist.set_title('Popularity of Game Genres', fontsize=14, weight='bold', pad=15)
            ax_dist.set_xlabel('Number of Players')
            ax_dist.set_ylabel('')
            for container in ax_dist.containers:
                ax_dist.bar_label(container, padding=3)

        elif dist_choice == "Age":
            sns.histplot(df['Age'], bins=25, kde=True, ax=ax_dist, 
                         color='#9b59b6', edgecolor='white', alpha=0.7)
            ax_dist.set_title('Player Age Distribution', fontsize=14, weight='bold', pad=15)
            ax_dist.set_xlabel('Age')
            ax_dist.set_ylabel('Frequency')

        elif dist_choice == "PlayTimeHours":
            sns.histplot(df['PlayTimeHours'], bins=25, kde=True, ax=ax_dist, 
                         color='#3498db', edgecolor='white', alpha=0.7)
            ax_dist.set_title('Play Time Hours Distribution', fontsize=14, weight='bold', pad=15)
            ax_dist.set_xlabel('Play Time (Hours)')
            ax_dist.set_ylabel('Frequency')

        sns.despine()
        fig_dist.tight_layout()
        st.pyplot(fig_dist)

        st.markdown("---")
        
        # --- VERSUS SECTION ---
        st.markdown("#### ⚔️ Interactive Feature vs Feature Explorer")
        st.write("Compare numerical features against Engagement Levels.")
        
        col_x, col_y = st.columns(2)
        with col_x:
            # Dropdown exactly matching ipynb X-axis
            x_axis = st.selectbox("Select X-Axis Feature:", ["EngagementLevel"])
        with col_y:
            # Dropdown exactly matching ipynb Y-axis
            y_axis = st.selectbox("Select Y-Axis Feature:", ["PlayTimeHours"])
            
        fig_vs = plt.figure(figsize=(10, 6))
        
        # Recreate exact Violin Plot from ipynb
        if x_axis == 'EngagementLevel' and y_axis == 'PlayTimeHours':
            sns.violinplot(
                data=df, x='EngagementLevel', y='PlayTimeHours', 
                hue='EngagementLevel', order=['Low', 'Medium', 'High'], 
                palette='pastel', inner='quartile', legend=False,
                linewidth=1.5
            )
            plt.title('Play Time Hours by Engagement Level', fontsize=16, weight='bold', pad=15)
            plt.xlabel('Engagement Level', fontsize=12)
            plt.ylabel('Play Time (Hours)', fontsize=12)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig_vs)

    with sub3:
        # Restored the exact Half-Masked Heatmap from ipynb 
        st.markdown("####  Correlation Heatmap")
        st.write("Correlation of target variable among each numerical column.")
        
        fig_corr, ax_corr = plt.subplots(figsize=(12, 9))
        numeric_cols_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
        corr_matrix = numeric_cols_df.corr()
        
        # Create mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot Heatmap mimicking ipynb style
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
    
    col_perf1, col_perf2 = st.columns([1, 1.2])
    
    with col_perf1:
        st.markdown(f"##### Detailed Report: `{selected_model_name}`")
        report_dict = detailed_reports[selected_model_name]
        report_df = pd.DataFrame(report_dict).transpose().drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
    with col_perf2:
        st.markdown("##### Overall Accuracy Comparison (All Models)")
        
        # Bar chart comparing all models
        fig_summary = px.bar(
            summary_df.sort_values("Accuracy", ascending=True), 
            x="Accuracy", 
            y="Model", 
            orientation='h', 
            text_auto='.2%', 
            color_discrete_sequence=['#6A0DAD']
        )
        fig_summary.update_layout(xaxis=dict(range=[0, 1]), showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_summary, use_container_width=True)

# ------------------------------------------
# TAB 3: Single Prediction Result 
# ------------------------------------------
with tab_pred:
    st.markdown("### Prediction Result")
    if predict_btn:
        input_data = pd.DataFrame([[age, gender, location, genre, play_time, in_purchases, 
                                    difficulty, sessions, avg_duration, player_level, achievements]], 
                                  columns=feature_cols)
        
        for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
            input_data[col] = le_dict[col].transform(input_data[col])
            
        input_scaled = scaler.transform(input_data)
        model = models_dict[selected_model_name]
        pred_encoded = model.predict(input_scaled)[0]
        prediction = le_dict['EngagementLevel'].inverse_transform([pred_encoded])[0]
        
        st.success(f"Prediction completed successfully using **{selected_model_name}**!")
        st.metric(label="Predicted Engagement Level", value=prediction)
        
        st.write("---")
        st.write("**Player Profile Summary Evaluated:**")
        st.write(f"- **Genre & Difficulty:** {genre} | {difficulty}")
        st.write(f"- **Activity:** {play_time} Hours/Week | {sessions} Sessions")
        st.write(f"- **Progression:** Level {player_level} | {achievements} Achievements")
    else:
        st.info("👈 Please enter player details in the sidebar and click 'Predict' to view results.")