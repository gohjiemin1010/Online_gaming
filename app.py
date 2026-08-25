import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. 页面基本设置
st.set_page_config(page_title="Online Gaming Behavior Dashboard", layout="wide")
st.title("🎮 Online Gaming Behavior Analysis & Prediction Dashboard")
st.markdown("Based on your Jupyter Notebook exploration and machine learning models.")

# 设置 Seaborn 风格（跟你 ipynb 里的设置一样）
sns.set_theme(style="white", context="notebook", font_scale=1.1)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 2. 读取数据 (对应 ipynb 第1个 cell)
@st.cache_data
def load_data():
    df = pd.read_csv('online_gaming_behavior_dataset.csv')
    return df

df = load_data()

# 3. 机器学习模型训练 (支持用户输入并预测)
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
    
    # 模型 1: Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    # 模型 2: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    return rf, lr, le_dict, scaler, X.columns

rf_model, lr_model, le_dict, scaler, feature_cols = train_models(df)

# ==========================================
# 侧边栏：用户输入与预测系统
# ==========================================
st.sidebar.header("🔮 玩家参与度预测系统")
selected_model_name = st.sidebar.selectbox("选择预测模型", ["Random Forest", "Logistic Regression"])

# 输入控件
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

if st.sidebar.button("开始预测玩家等级/参与度"):
    input_data = pd.DataFrame([[age, gender, location, genre, play_time, in_purchases, 
                                difficulty, sessions, avg_duration, player_level, achievements]], 
                              columns=feature_cols)
    
    for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
        input_data[col] = le_dict[col].transform(input_data[col])
        
    input_scaled = scaler.transform(input_data)
    model = rf_model if selected_model_name == "Random Forest" else lr_model
    pred_encoded = model.predict(input_scaled)[0]
    prediction = le_dict['EngagementLevel'].inverse_transform([pred_encoded])[0]
    
    st.sidebar.success(f"🎯 预测成功！该玩家的 EngagementLevel 为: **{prediction}**")


# ==========================================
# 主页面：展示基于 Jupyter Notebook 的 15 个图表
# ==========================================
st.markdown("---")
st.header("📈 数据探索与 15 个可视化图表展示")

# 标签页分类
t1, t2, t3 = st.tabs(["Part 1: 基础数据理解 (5图)", "Part 2: 玩家行为分布 (5图)", "Part 3: 深度关联分析 (5图)"])

with t1:
    st.subheader("数据集基础概况与变量分布")
    
    # 模拟 Jupyter 里的输出显示
    st.write(f"**Dataset Shape:** `{df.shape}`")
    st.write(f"**Columns:** `{df.columns.tolist()}`")
    
    col1, col2 = st.columns(2)
    with col1:
        # 图 1: Engagement Level 分布条形图 (对应你 ipynb 里的代码思路)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='EngagementLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette=['#ff9999','#66b3ff','#99ff99'], ax=ax, legend=False)
        ax.set_title('1. Distribution of Engagement Levels')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        st.pyplot(fig)
        
    with col2:
        # 图 2: 游戏类型热度统计
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, y='GameGenre', hue='GameGenre', palette='crest', ax=ax, legend=False)
        ax.set_title('2. Popularity of Game Genres')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        # 图 3: 性别分布
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Gender', hue='Gender', palette='Set2', ax=ax, legend=False)
        ax.set_title('3. Gender Distribution')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        st.pyplot(fig)
    with col4:
        # 图 4: 地区分布
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Location', hue='Location', palette='muted', ax=ax, legend=False)
        ax.set_title('4. Location Distribution')
        ax.bar_label(ax.containers[0], padding=3)
        sns.despine()
        st.pyplot(fig)

    # 图 5: 游戏难度分布
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='GameDifficulty', hue='GameDifficulty', palette='Blues', ax=ax, legend=False)
    ax.set_title('5. Game Difficulty Breakdown')
    ax.bar_label(ax.containers[0], padding=3)
    sns.despine()
    st.pyplot(fig)

with t2:
    st.subheader("核心数值变量分布 (直方图 + KDE)")
    
    col5, col6 = st.columns(2)
    with col5:
        # 图 6: 年龄分布
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Age'], bins=25, kde=True, color='#9b59b6', edgecolor='white', ax=ax)
        ax.set_title('6. Player Age Distribution')
        sns.despine()
        st.pyplot(fig)
    with col6:
        # 图 7: 游戏时间分布
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['PlayTimeHours'], bins=25, kde=True, color='#3498db', edgecolor='white', ax=ax)
        ax.set_title('7. Play Time Hours Distribution')
        sns.despine()
        st.pyplot(fig)

    col7, col8 = st.columns(2)
    with col7:
        # 图 8: 每周会话次数
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['SessionsPerWeek'], bins=20, kde=True, color='#e67e22', edgecolor='white', ax=ax)
        ax.set_title('8. Sessions Per Week Distribution')
        sns.despine()
        st.pyplot(fig)
    with col8:
        # 图 9: 平均会话时长
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['AvgSessionDurationMinutes'], bins=25, kde=True, color='#2ecc71', edgecolor='white', ax=ax)
        ax.set_title('9. Avg Session Duration Distribution')
        sns.despine()
        st.pyplot(fig)

    # 图 10: 玩家等级分布
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['PlayerLevel'], bins=30, kde=True, color='#e74c3c', edgecolor='white', ax=ax)
    ax.set_title('10. Player Level Distribution')
    sns.despine()
    st.pyplot(fig)

with t3:
    st.subheader("高级对比与多变量关联分析")
    
    col9, col10 = st.columns(2)
    with col9:
        # 图 11: 参与度 vs 游戏时长
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='EngagementLevel', y='PlayTimeHours', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='pastel', ax=ax, legend=False)
        ax.set_title('11. Play Time vs Engagement Level')
        sns.despine()
        st.pyplot(fig)
    with col10:
        # 图 12: 参与度 vs 玩家等级
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='EngagementLevel', y='PlayerLevel', hue='EngagementLevel', order=['Low', 'Medium', 'High'], palette='coolwarm', ax=ax, legend=False)
        ax.set_title('12. Player Level vs Engagement Level')
        sns.despine()
        st.pyplot(fig)

    col11, col12 = st.columns(2)
    with col11:
        # 图 13: 游戏类型 vs 游戏时长小提琴图 (对应 ipynb 的高级美化风格)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=df, x='GameGenre', y='PlayTimeHours', hue='GameGenre', palette='muted', inner='quartile', ax=ax, legend=False)
        ax.set_title('13. Play Time by Game Genre')
        plt.xticks(rotation=30)
        sns.despine()
        st.pyplot(fig)
    with col12:
        # 图 14: 成就解锁数 vs 玩家等级散点图
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df.sample(2000), x='PlayerLevel', y='AchievementsUnlocked', hue='EngagementLevel', alpha=0.7, ax=ax)
        ax.set_title('14. Level vs Achievements (Sampled)')
        sns.despine()
        st.pyplot(fig)

    # 图 15: 全局数值特征相关性热力图 (Seaborn 版本)
    st.markdown("##### 15. Correlation Heatmap (Full Numerical Features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
    corr_matrix = numeric_cols.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='vlag', fmt=".2f", linewidths=1, ax=ax, center=0)
    st.pyplot(fig)