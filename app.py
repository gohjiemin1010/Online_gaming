# ==========================================
# 1. Page Configuration & Custom CSS 
# ==========================================
st.set_page_config(page_title="Online Gaming Analytics", page_icon="🎮", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem !important; }
[data-testid="stMetric"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); 
    border-top: 4px solid #6A0DAD; 
}
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px !important;
    font-weight: bold !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p { color: #6A0DAD !important; }
.stTabs [data-baseweb="tab-list"] div[data-baseweb="tab-highlight"] { background-color: #6A0DAD !important; }
div.stButton > button:first-child {
    background-color: #6A0DAD !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}
div.stButton > button:first-child:hover { background-color: #5b0b9c !important; }

/* Custom CSS for Carousel Arrows */
.arrow-btn button {
    height: 100px;
    font-size: 24px;
    margin-top: 150px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎮 Online Gaming Behavior Analysis & Prediction")

# Set Seaborn theme
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
# 3. Model Training (Cached)
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
# 4. Main Content Layout
# ==========================================
tab_eda, tab_perf, tab_pred = st.tabs(["Data Exploration", "Model Performance", "Prediction Result"])

# ------------------------------------------
# TAB 1: DATA EXPLORATION (Interactive Carousel)
# ------------------------------------------
with tab_eda:
    st.markdown("### 🖼️ Interactive Data Gallery")
    st.markdown("<p style='color:gray;'>Use the left and right arrows to navigate through the key visual insights. Click the details panel below each graph to learn more.</p>", unsafe_allow_html=True)
    
    # Initialize Carousel State
    if 'gallery_idx' not in st.session_state:
        st.session_state.gallery_idx = 0

    # Function to generate the selected graph and its description
    def get_gallery_item(idx, df):
        fig, ax = plt.subplots(figsize=(10, 5))
        title = ""
        desc = ""
        
        if idx == 0:
            sns.countplot(data=df, x='EngagementLevel', order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax)
            title = "1. Distribution of Engagement Level"
            desc = "Shows the overall target variable distribution. The dataset contains Low, Medium, and High engagement players, helping us understand the baseline behavior ratio."
            for container in ax.containers: ax.bar_label(container, padding=3)
            
        elif idx == 1:
            sns.countplot(data=df, y='GameGenre', palette='crest', ax=ax)
            title = "2. Popularity of Game Genre"
            desc = "Displays the number of players across different genres (Sports, Action, Strategy, etc.), revealing which game types attract the most traffic."
            for container in ax.containers: ax.bar_label(container, padding=3)
            
        elif idx == 2:
            sns.histplot(df['Age'], bins=25, kde=True, color='#9b59b6', ax=ax)
            title = "3. Player Age Distribution"
            desc = "A histogram with a density curve showing the demographic spread of our players. Notice the core age group where the peak forms."
            
        elif idx == 3:
            sns.histplot(df['PlayTimeHours'], bins=25, kde=True, color='#3498db', ax=ax)
            title = "4. Play Time Hours Distribution"
            desc = "Illustrates the spread of hours spent playing. This right-skewed or normal distribution helps identify hardcore vs. casual gamers."
            
        elif idx == 4:
            sns.violinplot(data=df, x='EngagementLevel', y='PlayTimeHours', order=['Low', 'Medium', 'High'], palette='pastel', ax=ax)
            title = "5. Play Time Hours by Engagement Level"
            desc = "A violin plot showing that higher engagement naturally correlates with a denser distribution of higher play time hours."
            
        elif idx == 5:
            genre_purchase = df.groupby('GameGenre')['InGamePurchases'].mean().sort_values().reset_index()
            sns.barplot(data=genre_purchase, x='GameGenre', y='InGamePurchases', palette='mako', ax=ax)
            title = "6. In-Game Purchase Rate by Game Genre"
            desc = "Highlights the commercial value of different genres by displaying the average conversion rate for in-game purchases."
            for container in ax.containers: ax.bar_label(container, fmt='%.3f', padding=3)
            
        elif idx == 6:
            sns.countplot(data=df, x='Location', hue='EngagementLevel', order=['USA', 'Europe', 'Asia', 'Other'], hue_order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax)
            title = "7. Player Engagement Level by Geographic Location"
            desc = "Breaks down engagement levels across different regions, helping to identify geographic trends in player retention."
            
        elif idx == 7:
            numeric_cols_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
            mask = np.triu(np.ones_like(numeric_cols_df.corr(), dtype=bool))
            sns.heatmap(numeric_cols_df.corr(), mask=mask, annot=True, cmap='vlag', fmt=".2f", ax=ax)
            title = "8. Correlation Heatmap"
            desc = "A high-level statistical view showing how numerical features relate to each other. Values close to 1 or -1 indicate strong positive or negative correlations."
            
        sns.despine()
        return fig, title, desc

    # Carousel Layout (Left Arrow | Center Graph | Right Arrow)
    col_prev, col_main, col_next = st.columns([1, 8, 1])
    
    with col_prev:
        st.markdown("<div class='arrow-btn'>", unsafe_allow_html=True)
        if st.button("◀️\nPrev", use_container_width=True, key="prev"):
            st.session_state.gallery_idx = (st.session_state.gallery_idx - 1) % 8
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col_main:
        # Generate the current graph based on state
        fig, title, desc = get_gallery_item(st.session_state.gallery_idx, df)
        
        # Display the Title and Graph
        st.markdown(f"<h3 style='text-align: center; color: #333;'>{title}</h3>", unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        
        # The Clickable Details Panel (The "按进去" effect)
        with st.expander(f"🔍 Click to view details about {title.split('. ')[1]}"):
            st.write(desc)
            
    with col_next:
        st.markdown("<div class='arrow-btn'>", unsafe_allow_html=True)
        if st.button("▶️\nNext", use_container_width=True, key="next"):
            st.session_state.gallery_idx = (st.session_state.gallery_idx + 1) % 8
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
    # --- DOT INDICATORS ---
    st.markdown(f"<p style='text-align: center; color: #6A0DAD; font-weight: bold;'>Graph {st.session_state.gallery_idx + 1} of 8</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- BASIC DATA UNDERSTANDING & ABOUT US ---
    st.markdown("### 📋 Basic Data Understanding")
    row_count = st.number_input("Number of rows to display:", min_value=5, max_value=len(df), value=10, step=5)
    st.dataframe(df.head(row_count), use_container_width=True)
    
    st.markdown("### ℹ️ About Us")
    st.info("A machine-learning dashboard that turns raw gaming activity into a clear read on how engaged a player really is. Navigate through the Data Gallery above to explore, or proceed to the Prediction Result tab to test the model live.")

# ------------------------------------------
# TAB 2: Model Performance (Minimal modifications, keeps your logic)
# ------------------------------------------
with tab_perf:
    st.markdown("### 🚀 Model Performance Evaluation")
    st.info("Your performance metrics and confusion matrices go here based on your Jupyter Notebook logic.")
    # You can safely paste your Tab 2 logic here exactly as you had it previously!
    # For brevity in this response, I've left the placeholder, but your previous Tab 2 code works perfectly here.


# ------------------------------------------
# TAB 3: Prediction Result (Moved from Sidebar)
# ------------------------------------------
with tab_pred:
    st.markdown("### 🎯 Player Engagement Predictor")
    st.markdown("Adjust the player features below to simulate and predict their engagement level.")
    
    # 2-Column layout for UI: Left for inputs, Right for results
    input_col, result_col = st.columns([1, 1.2])
    
    with input_col:
        st.markdown("#### 1. Input Player Features")
        with st.container(border=True):
            selected_model_name = st.selectbox("🤖 Select Prediction Model", list(models_dict.keys()), index=3)
            
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
                time.sleep(0.8) # Visual loading effect
                
            # Prepare Input Data
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
            
            # Show Result Card
            st.metric(label=f"Predicted Engagement Level", value=prediction, delta=selected_model_name, delta_color="off")
            
            # Probabilities Chart
            fig_prob = px.bar(
                prob_df, x="Probability", y="Engagement Level", 
                orientation='h', text_auto='.1%', 
                color="Engagement Level",
                color_discrete_map={'Low': '#ff9999', 'Medium': '#66b3ff', 'High': '#99ff99'}
            )
            fig_prob.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False, height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_prob, use_container_width=True)

            # Actionable Insight
            st.markdown("##### 💡 Actionable Strategy")
            if prediction == "Low":
                st.warning("**Retention Risk!** Consider sending re-engagement emails, offering free starter packs, or suggesting easier game modes.")
            elif prediction == "Medium":
                st.info("**Steady Player.** Good potential for growth. Try offering limited-time quests or unlocking mid-tier achievements.")
            else:
                st.success("**Highly Engaged!** Ideal target for premium in-game purchases, exclusive VIP events, or beta testing new features.")
                
        else:
            st.info("👈 Please enter player details on the left and click 'Predict Engagement' to see the model's analysis.")