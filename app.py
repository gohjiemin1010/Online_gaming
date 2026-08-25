import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Page Configuration
st.set_page_config(page_title="Online Gaming Behavior Dashboard", layout="wide")

# Use markdown for a smaller, single-line title
st.markdown("### 🎮 Online Gaming Behavior Analysis & Prediction Dashboard")

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

# 3. Train Machine Learning Models & Calculate Metrics
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
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    metrics = {}
    target_names = le_dict['EngagementLevel'].classes_
    
    # Train all 4 models and save metrics (output_dict=True for table rendering later)
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test_scaled)
        trained_models[name] = model
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Report": classification_report(y_test, pred, target_names=target_names, output_dict=True)
        }
        
    return trained_models, le_dict, scaler, X.columns, metrics

models_dict, le_dict, scaler, feature_cols, model_metrics = train_models(df)

# ==========================================
# Sidebar: User Input & Prediction System
# ==========================================
st.sidebar.header("🔮 Player Engagement Predictor")
selected_model_name = st.sidebar.selectbox("Select Prediction Model", list(models_dict.keys()))

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
    
    # Predict using the selected model
    model = models_dict[selected_model_name]
    pred_encoded = model.predict(input_scaled)[0]
    prediction = le_dict['EngagementLevel'].inverse_transform([pred_encoded])[0]
    
    st.sidebar.success(f"🎯 Prediction Success! Predicted Engagement Level: **{prediction}**")

# Simplified Quick Metrics Expander in Sidebar
with st.sidebar.expander("📊 Quick Model Accuracy"):
    st.markdown(f"**Model:** {selected_model_name}")
    st.markdown(f"**Accuracy:** {model_metrics[selected_model_name]['Accuracy']:.2%}")

# ==========================================
# Main Page: Two Main Tabs Layout
# ==========================================
st.markdown("---")
main_tab1, main_tab2 = st.tabs(["📊 Data Exploration", "🤖 Model Performance"])

# ---------------------------------------------------------
# MAIN TAB 1: DATA EXPLORATION (The 15 Charts)
# ---------------------------------------------------------
with main_tab1:
    st.markdown("#### Data Exploration & Visualization")
    t1, t2, t3 = st.tabs(["Basic Data Understanding", "Player Behavior Distribution", "In-Depth Correlation Analysis"])

    with t1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='EngagementLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax, legend=False)
            ax.set_title('Distribution of Engagement Levels', pad=15)
            ax.bar_label(ax.containers[0], padding=3)
            ax.margins(y=0.2) # Adds 20% top margin to prevent overlap
            sns.despine()
            fig.tight_layout() 
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, y='GameGenre', hue='GameGenre', palette='crest', ax=ax, legend=False)
            ax.set_title('Popularity of Game Genres', pad=15)
            ax.bar_label(ax.containers[0], padding=3)
            ax.margins(x=0.2) # Adds right margin to prevent horizontal overlap
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

    with t2:
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

    with t3:
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

        st.markdown("##### Correlation Heatmap (Full Numerical Features)")
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
        corr_matrix = numeric_cols.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='vlag', fmt=".2f", linewidths=1, ax=ax, center=0)
        fig.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------
# MAIN TAB 2: MODEL PERFORMANCE (Metrics Table, Graph, Summary)
# ---------------------------------------------------------
with main_tab2:
    st.markdown(f"#### 📊 Performance Metrics for: **{selected_model_name}**")
    
    # 1. Convert classification report dict to DataFrame
    report_dict = model_metrics[selected_model_name]["Report"]
    report_df = pd.DataFrame(report_dict).transpose()
    
    col_table, col_graph = st.columns([1, 1.5])
    
    with col_table:
        st.markdown("**Classification Report Table**")
        # Display nicely formatted dataframe (excluding accuracy row for cleaner look)
        display_df = report_df.drop(['accuracy'], errors='ignore')
        st.dataframe(display_df.style.format("{:.3f}"), use_container_width=True)
        
    with col_graph:
        st.markdown("**Metrics Visualization (Precision, Recall, F1-Score)**")
        # Extract only the class metrics (High, Low, Medium) for plotting
        class_metrics = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        
        # Plotly grouped bar chart for precision, recall, f1-score
        fig_metrics = px.bar(
            class_metrics, 
            y=['precision', 'recall', 'f1-score'], 
            barmode='group', 
            text_auto='.2f',
            title=f"Class-wise Metrics - {selected_model_name}",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_metrics.update_layout(xaxis_title="Engagement Level Class", yaxis_title="Score", legend_title="Metrics")
        st.plotly_chart(fig_metrics, use_container_width=True)

    # 2. Summary of ALL Models
    st.markdown("---")
    st.markdown("#### 🏆 Summary of All Prediction Models")
    
    # Create DataFrame for all models' accuracy
    acc_data = [{"Model": name, "Accuracy": data["Accuracy"]} for name, data in model_metrics.items()]
    acc_df = pd.DataFrame(acc_data).sort_values(by="Accuracy", ascending=False)
    
    col_sum1, col_sum2 = st.columns([1, 2])
    
    with col_sum1:
        st.markdown("**Accuracy Comparison Table**")
        st.dataframe(acc_df.style.format({"Accuracy": "{:.2%}"}), use_container_width=True, hide_index=True)
        
    with col_sum2:
        # Bar chart comparing all models
        fig_summary = px.bar(
            acc_df, 
            x="Accuracy", 
            y="Model", 
            orientation='h', 
            text_auto='.2%', 
            color="Model",
            title="Overall Accuracy Comparison"
        )
        fig_summary.update_layout(xaxis=dict(range=[0, 1]), showlegend=False)
        st.plotly_chart(fig_summary, use_container_width=True)