import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="Online Gaming Analytics", page_icon="🎮", layout="wide")
st.markdown("## 🎮 Online Gaming Behavior Analysis & Prediction")

# Set Seaborn theme
sns.set_theme(style="white", context="notebook", font_scale=1.0)
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
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest (Champion)": RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    summary_metrics = []
    detailed_reports = {}
    target_names = le_dict['EngagementLevel'].classes_
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test_scaled)
        trained_models[name] = model
        
        # Calculate scores
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
        
        summary_metrics.append({
            "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1
        })
        
        detailed_reports[name] = classification_report(y_test, pred, target_names=target_names, output_dict=True)
        
    summary_df = pd.DataFrame(summary_metrics)
    
    return trained_models, le_dict, scaler, X.columns, summary_df, detailed_reports

models_dict, le_dict, scaler, feature_cols, summary_df, detailed_reports = train_models(df)

# ==========================================
# 4. Sidebar: User Input (Single Prediction)
# ==========================================
st.sidebar.markdown("### 🔮 Single Player Prediction")
selected_model_name = st.sidebar.selectbox("Select Model", list(models_dict.keys()), index=3)

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

predict_btn = st.sidebar.button("🚀 Predict Risk / Engagement", use_container_width=True)

# ==========================================
# 5. Main Content: 5 Professional Tabs
# ==========================================
tab_pred, tab_eda, tab_perf, tab_batch, tab_about = st.tabs([
    "🎯 Prediction Result", "📊 Data Exploration", "📈 Model Performance", "📁 Batch Prediction", "ℹ️ About"
])

# ------------------------------------------
# TAB 1: Single Prediction Result
# ------------------------------------------
with tab_pred:
    st.markdown("### Player Engagement Prediction")
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
        
        st.success("Prediction completed successfully!")
        
        # Display result nicely
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Predicted Engagement Level", value=prediction)
            st.info(f"Model Used: {selected_model_name}")
        with col_res2:
            st.write("**Player Profile Summary:**")
            st.write(f"- **Genre & Difficulty:** {genre} | {difficulty}")
            st.write(f"- **Activity:** {play_time} Hours/Week | {sessions} Sessions")
            st.write(f"- **Progression:** Level {player_level} | {achievements} Achievements")
    else:
        st.info("👈 Please enter player details in the sidebar and click 'Predict' to view results.")

# ------------------------------------------
# TAB 2: Data Exploration
# ------------------------------------------
with tab_eda:
    st.markdown("### Exploratory Data Analysis")
    
    # Dataset Overview Metrics (Like in the video)
    st.markdown("##### Dataset Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Players (Rows)", f"{df.shape[0]:,}")
    m2.metric("Total Features (Columns)", df.shape[1])
    m3.metric("Avg Play Time", f"{df['PlayTimeHours'].mean():.1f} hrs")
    m4.metric("Avg Player Level", f"{df['PlayerLevel'].mean():.0f}")
    
    st.markdown("---")
    
    # Sub-tabs for 15 charts
    sub1, sub2, sub3 = st.tabs(["Basic Data Understanding", "Player Behavior Distribution", "In-Depth Correlation Analysis"])

    with sub1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='EngagementLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax, legend=False)
            ax.set_title('Distribution of Engagement Levels', pad=15)
            ax.bar_label(ax.containers[0], padding=3)
            ax.margins(y=0.2) 
            sns.despine()
            fig.tight_layout() 
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, y='GameGenre', hue='GameGenre', palette='crest', ax=ax, legend=False)
            ax.set_title('Popularity of Game Genres', pad=15)
            ax.bar_label(ax.containers[0], padding=3)
            ax.margins(x=0.2) 
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='Gender', hue='Gender', palette='Set2', ax=ax, legend=False)
            ax.set_title('Gender Distribution', pad=15)
            ax.bar_label(ax.containers[0], padding=3)
            ax.margins(y=0.2)
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)
            
        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='Location', hue='Location', palette='muted', ax=ax, legend=False)
            ax.set_title('Location Distribution', pad=15)
            ax.bar_label(ax.containers[0], padding=3)
            ax.margins(y=0.2)
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x='GameDifficulty', hue='GameDifficulty', palette='Blues', ax=ax, legend=False)
        ax.set_title('Game Difficulty Breakdown', pad=15)
        ax.bar_label(ax.containers[0], padding=3)
        ax.margins(y=0.2)
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    with sub2:
        col5, col6 = st.columns(2)
        with col5:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['Age'], bins=25, kde=True, color='#9b59b6', edgecolor='white', ax=ax)
            ax.set_title('Player Age Distribution')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)
            
        with col6:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['PlayTimeHours'], bins=25, kde=True, color='#3498db', edgecolor='white', ax=ax)
            ax.set_title('Play Time Hours Distribution')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

        col7, col8 = st.columns(2)
        with col7:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['SessionsPerWeek'], bins=20, kde=True, color='#e67e22', edgecolor='white', ax=ax)
            ax.set_title('Sessions Per Week Distribution')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)
            
        with col8:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['AvgSessionDurationMinutes'], bins=25, kde=True, color='#2ecc71', edgecolor='white', ax=ax)
            ax.set_title('Avg Session Duration Distribution')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['PlayerLevel'], bins=30, kde=True, color='#e74c3c', edgecolor='white', ax=ax)
        ax.set_title('Player Level Distribution')
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)

    with sub3:
        col9, col10 = st.columns(2)
        with col9:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='EngagementLevel', y='PlayTimeHours', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='pastel', ax=ax, legend=False)
            ax.set_title('Play Time vs Engagement Level')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)
            
        with col10:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='EngagementLevel', y='PlayerLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='coolwarm', ax=ax, legend=False)
            ax.set_title('Player Level vs Engagement Level')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

        col11, col12 = st.columns(2)
        with col11:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.violinplot(data=df, x='GameGenre', y='PlayTimeHours', hue='GameGenre', palette='muted', inner='quartile', ax=ax, legend=False)
            ax.set_title('Play Time by Game Genre')
            plt.xticks(rotation=30)
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)
            
        with col12:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df.sample(2000), x='PlayerLevel', y='AchievementsUnlocked', hue='EngagementLevel', alpha=0.7, ax=ax)
            ax.set_title('Level vs Achievements (Sampled)')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("##### Correlation Heatmap (Numerical Features)")
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
        corr_matrix = numeric_cols.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='vlag', fmt=".2f", linewidths=1, ax=ax, center=0)
        fig.tight_layout()
        st.pyplot(fig)

# ------------------------------------------
# TAB 3: Model Performance
# ------------------------------------------
with tab_perf:
    st.markdown("### Model Performance Evaluation")
    
    # Overview Table for all models
    st.markdown("##### Performance Comparison (All Models)")
    # Apply styling to dataframe
    styled_df = summary_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"
    }).background_gradient(cmap='Greens', subset=['Accuracy', 'F1-Score'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col_perf1, col_perf2 = st.columns([1, 1.2])
    
    with col_perf1:
        st.markdown(f"##### Detailed Report: `{selected_model_name}`")
        report_dict = detailed_reports[selected_model_name]
        report_df = pd.DataFrame(report_dict).transpose().drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
    with col_perf2:
        st.markdown("##### Performance Comparison Graph")
        # Melting the summary dataframe for grouped bar chart
        melted_df = summary_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_perf = px.bar(
            melted_df, x="Metric", y="Score", color="Model", barmode="group",
            text_auto='.3f', color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_perf.update_layout(yaxis=dict(range=[0, 1]), margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig_perf, use_container_width=True)

# ------------------------------------------
# TAB 4: Batch Prediction
# ------------------------------------------
with tab_batch:
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file to predict engagement levels for multiple players at once.")
    st.info("Note: The CSV must contain the exact same columns as the training data (excluding PlayerID and EngagementLevel).")
    
    uploaded_file = st.file_uploader("Upload Player Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("Run Batch Prediction"):
            with st.spinner("Processing data and predicting..."):
                try:
                    # Filter only required columns
                    process_df = batch_df[feature_cols].copy()
                    
                    # Encode categorical
                    for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
                        process_df[col] = le_dict[col].transform(process_df[col])
                        
                    # Scale
                    process_scaled = scaler.transform(process_df)
                    
                    # Predict
                    model = models_dict[selected_model_name]
                    preds_encoded = model.predict(process_scaled)
                    preds = le_dict['EngagementLevel'].inverse_transform(preds_encoded)
                    
                    # Add prediction to original df
                    batch_df['Predicted_Engagement'] = preds
                    
                    st.success(f"Successfully generated predictions for {len(batch_df)} players!")
                    st.dataframe(batch_df[['Predicted_Engagement'] + feature_cols.tolist()], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing file. Please ensure columns match perfectly. Error: {e}")

# ------------------------------------------
# TAB 5: About This Project
# ------------------------------------------
with tab_about:
    st.markdown("### About This Project")
    st.markdown("""
    This application uses Machine Learning to predict **Player Engagement Levels** (Low, Medium, High) based on player behavior and demographics.
    
    #### ⚙️ 4 Base Models Evaluated:
    1. **Logistic Regression:** Used as a baseline linear model.
    2. **K-Nearest Neighbors (KNN):** Evaluates players based on similarity to others.
    3. **Decision Tree:** Uses branching logic to categorize players based on thresholds.
    4. **Random Forest:** An ensemble of decision trees, acting as the **Champion Model** due to its robustness against overfitting.
    
    #### 📂 Features Used:
    - **Demographics:** Age, Gender, Location
    - **Preferences:** Game Genre, Game Difficulty
    - **Activity:** Play Time (Hours), Sessions Per Week, Avg Session Duration
    - **Progression:** Player Level, Achievements Unlocked, In-Game Purchases
    
    *Disclaimer: This is for educational and portfolio purposes.*
    """)