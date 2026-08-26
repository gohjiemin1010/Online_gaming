import os
import glob
import re  
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

/* About Us Custom Feature Box */
.feature-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f8f9fa;
    border-left: 5px solid #6A0DAD;
    margin-bottom: 15px;
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
# 5. Main Content: Tabs Layout (Added 'About Us')
# ==========================================
tab_eda, tab_perf, tab_pred, tab_about = st.tabs([
    " Data Exploration", " Model Performance", " Prediction Result", " About Us"
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
        
        # --- VERSUS & NOTEBOOK EXPLORER SECTION ---
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

        # --- CORRELATION HEATMAP SECTION ---
        st.markdown("#### 🔗 Correlation Heatmap")
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
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(summary_df.style.format({"Accuracy": "{:.2%}"}), use_container_width=True)
    
    with col2:
        fig = px.bar(summary_df, x='Accuracy', y='Model', color='Model', 
                     title='Model Accuracy Comparison', text_auto='.2%', 
                     color_discrete_sequence=px.colors.qualitative.Purp)
        fig.update_layout(xaxis=dict(range=[0.7, 1.0]))
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### Detailed View")
    
    model_choice = st.selectbox("Select Model for Detailed Analysis:", list(models_dict.keys()), index=1)
    
    report_df = pd.DataFrame(detailed_reports[model_choice]).transpose()
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Classification Report: {model_choice}**")
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
        
    with c2:
        st.markdown(f"**Confusion Matrix: {model_choice}**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrices[model_choice], annot=True, fmt='d', cmap='Purples', 
                    xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
        ax_cm.set_ylabel('Actual')
        ax_cm.set_xlabel('Predicted')
        st.pyplot(fig_cm)

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
# TAB 4: About Us (New Addition)
# ------------------------------------------
with tab_about:
    st.markdown("###  About This Project")
    st.write("Welcome to the **Online Gaming Behavior Analytics** platform! This system is designed to help game developers, publishers, and marketers make data-driven decisions by accurately predicting player engagement levels.")
    
    st.markdown("---")
    
    st.markdown("#### 🧠 The Machine Learning Engine")
    st.write("To provide accurate predictions, we trained and evaluated four distinct machine learning models on gaming behavior data. The system dynamically processes your inputs and classifies the player's engagement into **Low, Medium, or High**.")
    
    # Grid of Models
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success("**XGBoost (Recommended)**\n\nOur most powerful model. It uses advanced gradient boosting decision trees for highly accurate and fast predictions.")
    with c2:
        st.info("**Random Forest**\n\nAn ensemble learning method that builds multiple decision trees to ensure robust, stable, and overfitting-resistant results.")
    with c3:
        st.warning("**Logistic Regression**\n\nA solid baseline statistical model that provides excellent interpretability for linear relationships in player data.")
    with c4:
        st.error("**K-Nearest Neighbors**\n\nA distance-based algorithm that evaluates and classifies a player based on the most similar players in the dataset.")
        
    st.markdown("---")
    
    st.markdown("#### ⚙️ Core System Features")
    
    # Custom styled HTML boxes for features
    st.markdown("""
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div class="feature-box" style="flex: 1; min-width: 250px;">
            <h4>📊 Interactive EDA</h4>
            <p>Visually explores complex gaming data, uncovering hidden trends in age, genres, and geographic locations right from your browser.</p>
        </div>
        <div class="feature-box" style="flex: 1; min-width: 250px;">
            <h4>⚡ Real-Time Predictions</h4>
            <p>Instantly profiles a player based on their unique metrics (like play time and achievements) and computes exact confidence probabilities.</p>
        </div>
        <div class="feature-box" style="flex: 1; min-width: 250px;">
            <h4>💡 Actionable Insights</h4>
            <p>Translates raw predictions into tangible business strategies, such as sending re-engagement emails to high-risk churn players.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><div style='text-align: center; color: gray;'>Designed for Modern Game Analytics & Player Retention Strategies 🕹️</div>", unsafe_allow_html=True)