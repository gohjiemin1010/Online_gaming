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

# Custom CSS for Purple Theme, 3D Cards, Larger Tabs, and Spacing
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
    
    # Sub-tabs for detailed EDA
    sub1, sub2, sub3 = st.tabs(["Basic Data Understanding", "Features Distribution", "In-Depth Correlation Analysis"])

    with sub1:
        # Dataset Preview using number_input
        st.markdown("#### 🔍 Dataset Preview")
        st.write("Use the +/- buttons or type a number to view more rows.")
        row_count = st.number_input("Number of rows to display:", min_value=5, max_value=len(df), value=100, step=10)
        st.dataframe(df.head(row_count), use_container_width=True)
        
        st.markdown("---")
        
        # Statistical Summaries Dropdown (From Jupyter Notebook)
        st.markdown("#### 📋 Statistical Summaries")
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
        st.markdown("#### 📊 Features Distribution")
        st.write("Select any feature (Numerical or Categorical) to view its distribution.")
        
        # Combine all relevant columns for the dropdown
        all_features = df.columns.drop(['PlayerID'], errors='ignore').tolist()
        dist_choice = st.selectbox("Select Feature to Visualize:", ["All"] + all_features)
        
        if dist_choice == "All":
            # Display all features in a grid
            grid_cols = st.columns(2)
            for i, col in enumerate(all_features):
                with grid_cols[i % 2]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    if df[col].dtype in ['int64', 'float64']:
                        sns.histplot(df[col], bins=25, kde=True, color='#6A0DAD', edgecolor='white', ax=ax)
                    else:
                        sns.countplot(data=df, x=col, palette='Purples', ax=ax)
                    ax.set_title(f'{col} Distribution', pad=15)
                    ax.margins(y=0.2)
                    sns.despine()
                    fig.tight_layout()
                    st.pyplot(fig)
        else:
            # Display single selected feature
            fig, ax = plt.subplots(figsize=(10, 5))
            if df[dist_choice].dtype in ['int64', 'float64']:
                sns.histplot(df[dist_choice], bins=25, kde=True, color='#6A0DAD', edgecolor='white', ax=ax)
            else:
                sns.countplot(data=df, x=dist_choice, palette='Purples', ax=ax)
            ax.set_title(f'{dist_choice} Distribution', pad=15)
            ax.margins(y=0.2)
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

    with sub3:
        st.markdown("#### 🔗 Advanced Correlation Analysis")
        col9, col10 = st.columns(2)
        with col9:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='EngagementLevel', y='PlayTimeHours', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='Purples', ax=ax, legend=False)
            ax.set_title('Play Time vs Engagement Level')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)
            
        with col10:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='EngagementLevel', y='PlayerLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='Purples', ax=ax, legend=False)
            ax.set_title('Player Level vs Engagement Level')
            sns.despine()
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        
        # Interactive Feature vs Feature Explorer
        st.markdown("#### ⚔️ Interactive Feature vs Feature Explorer")
        st.write("Select any two numerical features to see how they correlate with each other and player engagement.")
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('PlayerID', errors='ignore').tolist()
        col_x, col_y = st.columns(2)
        with col_x:
            x_axis = st.selectbox("Select X-Axis Feature:", num_cols, index=0)
        with col_y:
            y_axis = st.selectbox("Select Y-Axis Feature:", num_cols, index=len(num_cols)-1)
            
        fig_vs = px.scatter(df.sample(2000, random_state=42), x=x_axis, y=y_axis, color="EngagementLevel", 
                            title=f"Scatter Plot: {x_axis} vs {y_axis}", opacity=0.7,
                            color_discrete_sequence=['#6A0DAD', '#9b59b6', '#d2b4de']) # Purple theme
        st.plotly_chart(fig_vs, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🌡️ Correlation Heatmap (Numerical Features)")
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric_cols_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
        corr_matrix = numeric_cols_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='Purples', fmt=".2f", linewidths=1, ax=ax, center=0)
        fig.tight_layout()
        st.pyplot(fig)

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
        
        # Bar chart comparing all models (Purple Theme)
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
# TAB 3: Single Prediction Result (Moved to the end)
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