import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Page Configuration
st.set_page_config(page_title="Online Gaming Behavior Dashboard", layout="wide")
st.title("🎮 Online Gaming Behavior Analysis & Prediction Dashboard")
st.markdown("Based on Exploratory Data Analysis and Machine Learning Models.")

# Set Seaborn theme and remove top/right borders
sns.set_theme(style="white", context="notebook", font_scale=1.1)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 2. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('online_gaming_behavior_dataset.csv')
    return df

df = load_data()

# 3. Train Machine Learning Models
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
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    # Model 2: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    return rf, lr, le_dict, scaler, X.columns

rf_model, lr_model, le_dict, scaler, feature_cols = train_models(df)

# ==========================================
# Sidebar: User Input & Prediction System
# ==========================================
st.sidebar.header("🔮 Player Engagement Predictor")
selected_model_name = st.sidebar.selectbox("Select Prediction Model", ["Random Forest", "Logistic Regression"])

# Input widgets
age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 25)
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
location = st.sidebar.selectbox("Location", df['Location'].unique())
genre = st.sidebar.selectbox("Game Genre", df['GameGenre'].unique())
play_time = st.sidebar.number_input("Play Time (Hours)", 0.0, 24.0, 10.0)
in_purchases = st.sidebar.selectbox("In-Game Purchases", [0, 1])
difficulty = st.sidebar.selectbox("Game Difficulty", df['GameDifficulty'].unique())
sessions = st.sidebar.slider("Sessions Per Week", int(df['SessionsPerWeek'].min()), int(df['SessionsPerWeek'].max()), 5)
avg_duration = st.sidebar.slider("Avg Session Duration (Minutes)", int(df['AvgSessionDurationMinutes'].min()), int(df['AvgSessionDurationMinutes'].max()), 60)
player_level = st.sidebar.slider("Player Level", int(df['PlayerLevel'].min()), int(df['PlayerLevel'].max()), 30)
achievements = st.sidebar.slider("Achievements Unlocked", int(df['AchievementsUnlocked'].min()), int(df['AchievementsUnlocked'].max()), 15)

if st.sidebar.button("Predict Engagement Level"):
    input_data = pd.DataFrame([[age, gender, location, genre, play_time, in_purchases, 
                                difficulty, sessions, avg_duration, player_level, achievements]], 
                              columns=feature_cols)
    
    for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
        input_data[col] = le_dict[col].transform(input_data[col])
        
    input_scaled = scaler.transform(input_data)
    model = rf_model if selected_model_name == "Random Forest" else lr_model
    pred_encoded = model.predict(input_scaled)[0]
    prediction = le_dict['EngagementLevel'].inverse_transform([pred_encoded])[0]
    
    st.sidebar.success(f"🎯 Prediction Success! Predicted Engagement Level: **{prediction}**")

# ==========================================
# Main Page: Data Visualizations (15 Charts)
# ==========================================
st.markdown("---")
st.header("📈 Data Exploration & Visualization")

# Tabs for organization
t1, t2, t3 = st.tabs(["Basic Data Understanding", "Player Behavior Distribution", "In-Depth Correlation Analysis"])

with t1:
    st.subheader("Basic Dataset Overview & Categorical Variables")
    
    col1, col2 = st.columns(2)
    with col1:
        # Chart 1
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='EngagementLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax, legend=False)
        ax.set_title('Distribution of Engagement Levels')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        fig.tight_layout() # This prevents overlapping
        st.pyplot(fig)
        
    with col2:
        # Chart 2
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, y='GameGenre', hue='GameGenre', palette='crest', ax=ax, legend=False)
        ax.set_title('Popularity of Game Genres')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        # Chart 3
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Gender', hue='Gender', palette='Set2', ax=ax, legend=False)
        ax.set_title('Gender Distribution')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        
    with col4:
        # Chart 4
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Location', hue='Location', palette='muted', ax=ax, legend=False)
        ax.set_title('Location Distribution')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 5
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='GameDifficulty', hue='GameDifficulty', palette='Blues', ax=ax, legend=False)
    ax.set_title('Game Difficulty Breakdown')
    ax.bar_label(ax.containers[0], padding=3)
    sns.despine()
    fig.tight_layout()
    st.pyplot(fig)

with t2:
    st.subheader("Numerical Variables Distribution (Histogram + KDE)")
    
    col5, col6 = st.columns(2)
    with col5:
        # Chart 6
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Age'], bins=25, kde=True, color='#9b59b6', edgecolor='white', ax=ax)
        ax.set_title('Player Age Distribution')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        
    with col6:
        # Chart 7
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['PlayTimeHours'], bins=25, kde=True, color='#3498db', edgecolor='white', ax=ax)
        ax.set_title('Play Time Hours Distribution')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    col7, col8 = st.columns(2)
    with col7:
        # Chart 8
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['SessionsPerWeek'], bins=20, kde=True, color='#e67e22', edgecolor='white', ax=ax)
        ax.set_title('Sessions Per Week Distribution')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        
    with col8:
        # Chart 9
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['AvgSessionDurationMinutes'], bins=25, kde=True, color='#2ecc71', edgecolor='white', ax=ax)
        ax.set_title('Avg Session Duration Distribution')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 10
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['PlayerLevel'], bins=30, kde=True, color='#e74c3c', edgecolor='white', ax=ax)
    ax.set_title('Player Level Distribution')
    sns.despine()
    fig.tight_layout()
    st.pyplot(fig)

with t3:
    st.subheader("Advanced Correlation & Cross-Variable Analysis")
    
    col9, col10 = st.columns(2)
    with col9:
        # Chart 11
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='EngagementLevel', y='PlayTimeHours', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='pastel', ax=ax, legend=False)
        ax.set_title('Play Time vs Engagement Level')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        
    with col10:
        # Chart 12
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='EngagementLevel', y='PlayerLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='coolwarm', ax=ax, legend=False)
        ax.set_title('Player Level vs Engagement Level')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    col11, col12 = st.columns(2)
    with col11:
        # Chart 13
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=df, x='GameGenre', y='PlayTimeHours', hue='GameGenre', palette='muted', inner='quartile', ax=ax, legend=False)
        ax.set_title('Play Time by Game Genre')
        plt.xticks(rotation=30)
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        
    with col12:
        # Chart 14
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df.sample(2000), x='PlayerLevel', y='AchievementsUnlocked', hue='EngagementLevel', alpha=0.7, ax=ax)
        ax.set_title('Level vs Achievements (Sampled)')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    # Chart 15: Heatmap
    st.markdown("##### Correlation Heatmap (Full Numerical Features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
    corr_matrix = numeric_cols.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='vlag', fmt=".2f", linewidths=1, ax=ax, center=0)
    fig.tight_layout()
    st.pyplot(fig)