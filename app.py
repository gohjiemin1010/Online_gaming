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
    
    # Initialize 4 Models (Including XGBoost)
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
st.sidebar.markdown("### 🔮 Player Engagement Predictor")
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

predict_btn = st.sidebar.button("🚀 Predict Engagement", use_container_width=True)

# Quick Metrics in Sidebar
with st.sidebar.expander("📊 Quick Model Accuracy"):
    current_acc = summary_df[summary_df['Model'] == selected_model_name]['Accuracy'].values[0]
    st.markdown(f"**Model:** {selected_model_name}")
    st.markdown(f"**Accuracy:** {current_acc:.2%}")

# ==========================================
# 5. Main Content: Tabs Layout
# ==========================================
tab_pred, tab_eda, tab_perf = st.tabs([
    "🎯 Prediction Result", "📊 Data Exploration", "📈 Model Performance"
])

# ------------------------------------------
# TAB 1: Single Prediction Result
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
        
        st.success("Prediction completed successfully!")
        st.metric(label="Predicted Engagement Level", value=prediction)
        st.info(f"Model Used: {selected_model_name}")
    else:
        st.info("👈 Please enter player details in the sidebar and click 'Predict' to view results.")

# ------------------------------------------
# TAB 2: Data Exploration
# ------------------------------------------
with tab_eda:
    st.markdown("### Exploratory Data Analysis")
    
    # Beautiful Boxed Metric Cards
    st.markdown("##### Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        with st.container(border=True):
            st.metric("Total Players", f"{df.shape[0]:,}")
    with m2:
        with st.container(border=True):
            st.metric("Total Features", df.shape[1])
    with m3:
        with st.container(border=True):
            st.metric("Avg Play Time", f"{df['PlayTimeHours'].mean():.1f} hrs")
    with m4:
        with st.container(border=True):
            st.metric("Avg Player Level", f"{df['PlayerLevel'].mean():.0f}")
    with m5:
        with st.container(border=True):
            freq_eng = df['EngagementLevel'].mode()[0]
            st.metric("Most Frequent Engagement", freq_eng)
    
    st.markdown("---")
    
    # Sub-tabs for detailed EDA
    sub1, sub2, sub3 = st.tabs(["Basic Data Understanding", "Player Behavior Distribution", "In-Depth Correlation Analysis"])

    with sub1:
        # Dataset Preview Dropdown
        st.markdown("#### 🔍 Dataset Preview")
        row_count = st.selectbox("Select number of rows to display:", [5, 10, 20, 50, 100])
        st.dataframe(df.head(row_count), use_container_width=True)
        
        st.markdown("---")
        
        # Statistical Summaries Dropdown (From Jupyter Notebook)
        st.markdown("#### 📋 Statistical Summaries")
        summary_choice = st.selectbox("Select Summary Type:", ["Numerical Summary", "Categorical Summary"])
        
        if summary_choice == "Numerical Summary":
            st.markdown("**Full Dataset Statistical Profile (Numerical)**")
            # Mimicking the Jupyter Notebook logic
            num_desc = df.describe().T
            num_desc['range'] = num_desc['max'] - num_desc['min']
            num_desc['cv'] = (num_desc['std'] / num_desc['mean'] * 100).round(1)
            display_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'cv']
            st.dataframe(num_desc[display_cols].style.format("{:.2f}"), use_container_width=True)
            
        elif summary_choice == "Categorical Summary":
            st.markdown("**Categorical Features Value Counts**")
            cat_cols = df.select_dtypes(include=['object']).columns
            # Display categorical summaries as clean side-by-side tables
            table_cols = st.columns(len(cat_cols))
            for i, col in enumerate(cat_cols):
                with table_cols[i]:
                    st.markdown(f"**{col}**")
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, 'Count']
                    st.dataframe(vc, hide_index=True, use_container_width=True)

    with sub2:
        st.markdown("#### 📊 Numerical Features Distribution")
        # Dropdown to select a specific numerical column or "All"
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('PlayerID', errors='ignore').tolist()
        dist_choice = st.selectbox("Select Feature to Visualize:", ["All"] + num_cols)
        
        if dist_choice == "All":
            # Show all numerical distributions in a grid
            grid_cols = st.columns(2)
            for i, col in enumerate(num_cols):
                with grid_cols[i % 2]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df[col], bins=25, kde=True, color='#3498db', edgecolor='white', ax=ax)
                    ax.set_title(f'{col} Distribution', pad=15)
                    sns.despine()
                    fig.tight_layout()
                    st.pyplot(fig)
        else:
            # Show only the selected column
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[dist_choice], bins=25, kde=True, color='#9b59b6', edgecolor='white', ax=ax)
            ax.set_title(f'{dist_choice} Distribution', pad=15)
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

    with sub3:
        st.markdown("#### 🔗 Advanced Correlation Analysis")
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
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_summary.update_layout(xaxis=dict(range=[0, 1]), showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_summary, use_container_width=True)