import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
import time
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Page Configuration & Global Settings
# ==========================================
st.set_page_config(page_title="Online Gaming Analytics", page_icon="🎮", layout="wide")

# Set Seaborn theme
sns.set_theme(style="white", context="notebook", font_scale=1.1)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ==========================================
# 2. Advanced CSS (Fixed 3D Cards, Transparent Slider)
# ==========================================
st.markdown("""
<style>
.block-container { padding-top: 1.5rem !important; }

/* 3D Metric Cards styling with hover effect */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    padding: 15px 20px !important;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1) !important; 
    border-top: 4px solid #6A0DAD !important; 
    transition: all 0.3s ease-in-out !important; 
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px) !important; 
    box-shadow: 0px 10px 20px rgba(106, 13, 173, 0.2) !important; 
}

/* 3D Coverflow Slider Styling (Transparent Background) */
.slider-container {
    position: relative;
    width: 100%;
    height: 450px;
    display: flex;
    justify-content: center;
    align-items: center;
    perspective: 1200px;
    overflow: hidden;
    background: transparent; /* Removed gray box */
    margin-bottom: 20px;
}
.slider-card {
    position: absolute;
    width: 600px;
    height: 380px;
    transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
    border-radius: 15px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    background-color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;
}
.slider-card img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

/* 3D Position Classes */
.card-center {
    transform: translateX(0) translateZ(0) scale(1);
    z-index: 10;
    opacity: 1;
}
.card-left-1 {
    transform: translateX(-45%) translateZ(-150px) rotateY(15deg) scale(0.85);
    z-index: 5;
    opacity: 0.7;
}
.card-right-1 {
    transform: translateX(45%) translateZ(-150px) rotateY(-15deg) scale(0.85);
    z-index: 5;
    opacity: 0.7;
}
.card-left-2 {
    transform: translateX(-80%) translateZ(-300px) rotateY(25deg) scale(0.7);
    z-index: 4;
    opacity: 0.4;
}
.card-right-2 {
    transform: translateX(80%) translateZ(-300px) rotateY(-25deg) scale(0.7);
    z-index: 4;
    opacity: 0.4;
}
.card-hidden {
    transform: translateX(0) translateZ(-500px) scale(0.5);
    z-index: 1;
    opacity: 0;
}

/* Streamlit Native UI Overrides */
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px !important;
    font-weight: bold !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p { color: #6A0DAD !important; }
.stTabs [data-baseweb="tab-list"] div[data-baseweb="tab-highlight"] { background-color: #6A0DAD !important; }

div.stButton > button {
    background-color: #6A0DAD !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    width: 100%;
}
div.stButton > button:hover { background-color: #5b0b9c !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("##  Online Gaming Behavior Analytics")

# ==========================================
# 3. Data Loading & Graph Generation (Cached)
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv('online_gaming_behavior_dataset.csv')

df = load_data()

# Helper function to convert matplotlib figures to Base64 HTML strings
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=120, transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

@st.cache_data
def generate_gallery_assets(df):
    images_b64 = []
    titles = [
        "1. Distribution of Engagement Level",
        "2. Popularity of Game Genre",
        "3. Player Age Distribution",
        "4. Play Time Hours Distribution",
        "5. Play Time Hours by Engagement Level",
        "6. In-Game Purchase Rate by Game Genre",
        "7. Player Engagement Level by Geographic Location",
        "8. Correlation Heatmap"
    ]
    details = [
        "Shows the target variable distribution. The dataset is balanced across Low, Medium, and High engagement players, providing a solid baseline for our ML predictions.",
        "Displays the volume of players across different genres (Sports, Action, Strategy, etc.), revealing which game types drive the most traffic.",
        "A density histogram representing the demographic spread. This highlights the core age groups making up our player base.",
        "Illustrates the spread of play hours. The distribution helps identify the threshold between casual gamers and hardcore gamers.",
        "A violin plot confirming that higher engagement levels naturally correlate with a denser distribution of higher play time hours.",
        "Highlights commercial value by genre. It displays the average conversion rate (percentage) for in-game purchases.",
        "Breaks down engagement levels across different geographical regions, useful for identifying regional retention strengths.",
        "A high-level statistical matrix showing how numerical features relate. Values close to 1 or -1 indicate strong correlations."
    ]
    
    # 1. Engagement Level
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='EngagementLevel', order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax)
    ax.set_title(titles[0], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    # 2. Game Genre
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, y='GameGenre', palette='crest', ax=ax)
    ax.set_title(titles[1], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    # 3. Age Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Age'], bins=25, kde=True, color='#9b59b6', ax=ax)
    ax.set_title(titles[2], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    # 4. Play Time
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['PlayTimeHours'], bins=25, kde=True, color='#3498db', ax=ax)
    ax.set_title(titles[3], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    # 5. Play Time by Engagement
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x='EngagementLevel', y='PlayTimeHours', order=['Low', 'Medium', 'High'], palette='pastel', ax=ax)
    ax.set_title(titles[4], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    # 6. Purchase Rate by Genre
    fig, ax = plt.subplots(figsize=(8, 5))
    genre_purchase = df.groupby('GameGenre')['InGamePurchases'].mean().sort_values().reset_index()
    sns.barplot(data=genre_purchase, x='GameGenre', y='InGamePurchases', palette='mako', ax=ax)
    ax.set_title(titles[5], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    # 7. Engagement by Location
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Location', hue='EngagementLevel', order=['USA', 'Europe', 'Asia', 'Other'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax)
    ax.set_title(titles[6], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    # 8. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    numeric_cols_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
    mask = np.triu(np.ones_like(numeric_cols_df.corr(), dtype=bool))
    sns.heatmap(numeric_cols_df.corr(), mask=mask, annot=True, cmap='vlag', fmt=".2f", ax=ax)
    ax.set_title(titles[7], weight='bold')
    images_b64.append(fig_to_base64(fig))
    
    return images_b64, titles, details

images_b64, graph_titles, graph_details = generate_gallery_assets(df)

# ==========================================
# 4. Models Setup
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
    
    models = {"XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models, le_dict, scaler, X.columns

models_dict, le_dict, scaler, feature_cols = train_models(df)

# ==========================================
# 5. Main Tabs Layout
# ==========================================
tab_eda, tab_perf, tab_pred = st.tabs(["🖼️ Data Analysis", "📊 Model Performance", "🎯 Prediction Result"])

# ------------------------------------------
# TAB 1: DATA ANALYSIS
# ------------------------------------------
with tab_eda:
    
    # Dataset Overview Cards
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
    
    # Initialize State
    if 'gallery_idx' not in st.session_state:
        st.session_state.gallery_idx = 0
    total_cards = 8
    idx = st.session_state.gallery_idx

    # Calculate CSS classes for 3D positioning
    classes = ['card-hidden'] * total_cards
    classes[idx] = 'card-center'
    classes[(idx - 1) % total_cards] = 'card-left-1'
    classes[(idx - 2) % total_cards] = 'card-left-2'
    classes[(idx + 1) % total_cards] = 'card-right-1'
    classes[(idx + 2) % total_cards] = 'card-right-2'

    # Inject HTML for 3D Carousel
    html_carousel = f"""
    <div class="slider-container">
        <div class="slider-card {classes[0]}"><img src="data:image/png;base64,{images_b64[0]}"></div>
        <div class="slider-card {classes[1]}"><img src="data:image/png;base64,{images_b64[1]}"></div>
        <div class="slider-card {classes[2]}"><img src="data:image/png;base64,{images_b64[2]}"></div>
        <div class="slider-card {classes[3]}"><img src="data:image/png;base64,{images_b64[3]}"></div>
        <div class="slider-card {classes[4]}"><img src="data:image/png;base64,{images_b64[4]}"></div>
        <div class="slider-card {classes[5]}"><img src="data:image/png;base64,{images_b64[5]}"></div>
        <div class="slider-card {classes[6]}"><img src="data:image/png;base64,{images_b64[6]}"></div>
        <div class="slider-card {classes[7]}"><img src="data:image/png;base64,{images_b64[7]}"></div>
    </div>
    """
    st.markdown(html_carousel, unsafe_allow_html=True)

    # Navigation & Details Logic (Buttons moved closer to the center)
    col_space_left, col_prev, col_details, col_next, col_space_right = st.columns([1.5, 1, 4, 1, 1.5])
    
    with col_prev:
        st.write("") # Tiny spacer to push button down aligning with expander
        if st.button("◀ PREV", use_container_width=True):
            st.session_state.gallery_idx = (st.session_state.gallery_idx - 1) % total_cards
            st.rerun()
            
    with col_details:
        with st.expander(f"🔍 VIEW GRAPH DETAILS: {graph_titles[idx].split('. ')[1]}", expanded=False):
            st.markdown(f"**Description:**<br>{graph_details[idx]}", unsafe_allow_html=True)
            
    with col_next:
        st.write("") # Tiny spacer
        if st.button("NEXT ▶", use_container_width=True):
            st.session_state.gallery_idx = (st.session_state.gallery_idx + 1) % total_cards
            st.rerun()

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

# ------------------------------------------
# TAB 2: Model Performance
# ------------------------------------------
with tab_perf:
    st.markdown("### 🚀 Model Performance Evaluation")
    st.info("Your performance metrics and confusion matrices go here based on your Jupyter Notebook logic.")

# ------------------------------------------
# TAB 3: Prediction Result
# ------------------------------------------
with tab_pred:
    st.markdown("### 🎯 Player Engagement Predictor")
    st.markdown("Adjust the player features below to simulate and predict their engagement level.")
    
    input_col, result_col = st.columns([1, 1.2])
    
    with input_col:
        st.markdown("#### 1. Input Player Features")
        with st.container(border=True):
            # FIXED ERROR HERE: Changed index=3 to index=0
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
            
            predict_btn = st.button("🔮 Predict Engagement", use_container_width=True)

    with result_col:
        st.markdown("#### 2. Prediction Insights")
        if predict_btn:
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
            
            st.metric(label=f"Predicted Engagement Level", value=prediction, delta=selected_model_name, delta_color="off")
            
            fig_prob = px.bar(
                prob_df, x="Probability", y="Engagement Level", 
                orientation='h', text_auto='.1%', 
                color="Engagement Level",
                color_discrete_map={'Low': '#ff9999', 'Medium': '#66b3ff', 'High': '#99ff99'}
            )
            fig_prob.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False, height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown("##### 💡 Actionable Strategy")
            if prediction == "Low":
                st.warning("**Retention Risk!** Consider sending re-engagement emails, offering free starter packs, or suggesting easier game modes.")
            elif prediction == "Medium":
                st.info("**Steady Player.** Good potential for growth. Try offering limited-time quests or unlocking mid-tier achievements.")
            else:
                st.success("**Highly Engaged!** Ideal target for premium in-game purchases, exclusive VIP events, or beta testing new features.")
                
        else:
            st.info("👈 Please enter player details on the left and click 'Predict Engagement' to see the model's analysis.")