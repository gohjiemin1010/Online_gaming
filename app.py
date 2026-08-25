import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. 页面设置
st.set_page_config(page_title="Gaming Behavior Dashboard", layout="wide")
st.title("🎮 玩家游戏行为与参与度分析预测看板")

# 2. 读取数据 (使用缓存加速)
@st.cache_data
def load_data():
    df = pd.read_csv('online_gaming_behavior_dataset.csv')
    return df

df = load_data()

# 3. 数据预处理与模型训练 (使用缓存避免每次刷新重新训练)
@st.cache_resource
def train_models(df):
    df_model = df.copy()
    
    # 编码分类变量
    le_dict = {}
    cat_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le
        
    X = df_model.drop(['PlayerID', 'EngagementLevel'], axis=1)
    y = df_model['EngagementLevel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # 训练模型 1: 随机森林
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 训练模型 2: 逻辑回归
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    return rf_model, lr_model, le_dict, scaler, X.columns

rf_model, lr_model, le_dict, scaler, feature_cols = train_models(df)

# ==========================================
# 侧边栏：用户输入与预测功能
# ==========================================
st.sidebar.header("🔮 玩家参与度预测器")

# 模型选择
selected_model = st.sidebar.selectbox("选择预测模型", ["Random Forest (推荐)", "Logistic Regression"])

# 用户输入各项资料
age = st.sidebar.slider("年龄 (Age)", int(df['Age'].min()), int(df['Age'].max()), 25)
gender = st.sidebar.selectbox("性别 (Gender)", df['Gender'].unique())
location = st.sidebar.selectbox("地区 (Location)", df['Location'].unique())
genre = st.sidebar.selectbox("游戏类型 (Game Genre)", df['GameGenre'].unique())
play_time = st.sidebar.number_input("游戏时长/周 (PlayTimeHours)", min_value=0.0, max_value=168.0, value=10.0)
in_game_purchases = st.sidebar.selectbox("游戏内购买 (InGamePurchases)", [0, 1])
difficulty = st.sidebar.selectbox("游戏难度 (GameDifficulty)", df['GameDifficulty'].unique())
sessions = st.sidebar.slider("每周会话次数 (SessionsPerWeek)", int(df['SessionsPerWeek'].min()), int(df['SessionsPerWeek'].max()), 5)
avg_duration = st.sidebar.slider("平均会话时长/分钟 (AvgSessionDurationMinutes)", int(df['AvgSessionDurationMinutes'].min()), int(df['AvgSessionDurationMinutes'].max()), 60)
player_level = st.sidebar.slider("玩家等级 (PlayerLevel)", int(df['PlayerLevel'].min()), int(df['PlayerLevel'].max()), 50)
achievements = st.sidebar.slider("解锁成就数 (AchievementsUnlocked)", int(df['AchievementsUnlocked'].min()), int(df['AchievementsUnlocked'].max()), 20)

if st.sidebar.button("🚀 预测参与度等级"):
    # 整理输入数据
    input_data = pd.DataFrame([[age, gender, location, genre, play_time, in_game_purchases, 
                                difficulty, sessions, avg_duration, player_level, achievements]], 
                              columns=feature_cols)
    
    # 处理分类变量
    input_data['Gender'] = le_dict['Gender'].transform(input_data['Gender'])
    input_data['Location'] = le_dict['Location'].transform(input_data['Location'])
    input_data['GameGenre'] = le_dict['GameGenre'].transform(input_data['GameGenre'])
    input_data['GameDifficulty'] = le_dict['GameDifficulty'].transform(input_data['GameDifficulty'])
    
    # 标准化
    input_scaled = scaler.transform(input_data)
    
    # 预测
    model = rf_model if selected_model == "Random Forest (推荐)" else lr_model
    prediction_encoded = model.predict(input_scaled)[0]
    prediction = le_dict['EngagementLevel'].inverse_transform([prediction_encoded])[0]
    
    st.sidebar.success(f"🎯 预测结果：该玩家的参与度为 **{prediction}**")

# ==========================================
# 主页面：15 个数据可视化图表 (Dashboard)
# ==========================================
st.markdown("---")
st.header("📊 数据可视化看板 (共 15 个图表)")

# 使用 Tabs 来分类图表，让界面看起来更整洁
tab1, tab2, tab3 = st.tabs(["📌 基础分布 (6图)", "📈 深度分析 (6图)", "🔥 相关性与高级图表 (3图)"])

with tab1:
    col1, col2, col3 = st.columns(3)
    # 图 1
    with col1:
        fig1 = px.histogram(df, x="EngagementLevel", color="EngagementLevel", title="1. 参与度分布")
        st.plotly_chart(fig1, use_container_width=True)
    # 图 2
    with col2:
        fig2 = px.pie(df, names="GameGenre", title="2. 游戏类型占比", hole=0.3)
        st.plotly_chart(fig2, use_container_width=True)
    # 图 3
    with col3:
        fig3 = px.histogram(df, x="Location", color="Location", title="3. 玩家地区分布")
        st.plotly_chart(fig3, use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    # 图 4
    with col4:
        fig4 = px.pie(df, names="Gender", title="4. 性别比例")
        st.plotly_chart(fig4, use_container_width=True)
    # 图 5
    with col5:
        fig5 = px.histogram(df, x="GameDifficulty", color="GameDifficulty", title="5. 游戏难度偏好")
        st.plotly_chart(fig5, use_container_width=True)
    # 图 6
    with col6:
        fig6 = px.pie(df, names="InGamePurchases", title="6. 游戏内购买率")
        st.plotly_chart(fig6, use_container_width=True)

with tab2:
    col7, col8 = st.columns(2)
    # 图 7
    with col7:
        fig7 = px.histogram(df, x="Age", nbins=30, title="7. 年龄分布")
        st.plotly_chart(fig7, use_container_width=True)
    # 图 8
    with col8:
        fig8 = px.box(df, x="EngagementLevel", y="PlayTimeHours", color="EngagementLevel", title="8. 参与度 vs 游戏时长")
        st.plotly_chart(fig8, use_container_width=True)
        
    col9, col10 = st.columns(2)
    # 图 9
    with col9:
        fig9 = px.histogram(df, x="SessionsPerWeek", title="9. 每周会话次数分布")
        st.plotly_chart(fig9, use_container_width=True)
    # 图 10
    with col10:
        fig10 = px.box(df, x="EngagementLevel", y="PlayerLevel", color="EngagementLevel", title="10. 参与度 vs 玩家等级")
        st.plotly_chart(fig10, use_container_width=True)

    col11, col12 = st.columns(2)
    # 图 11
    with col11:
        fig11 = px.histogram(df, x="AvgSessionDurationMinutes", title="11. 平均会话时长")
        st.plotly_chart(fig11, use_container_width=True)
    # 图 12
    with col12:
        fig12 = px.histogram(df, x="AchievementsUnlocked", title="12. 解锁成就数量分布")
        st.plotly_chart(fig12, use_container_width=True)

with tab3:
    col13, col14 = st.columns(2)
    # 图 13
    with col13:
        fig13 = px.scatter(df, x="PlayTimeHours", y="PlayerLevel", color="EngagementLevel", title="13. 时长与等级散点图")
        st.plotly_chart(fig13, use_container_width=True)
    # 图 14
    with col14:
        fig14 = px.violin(df, x="GameGenre", y="PlayTimeHours", color="GameGenre", title="14. 游戏类型 vs 时长分布")
        st.plotly_chart(fig14, use_container_width=True)
        
    # 图 15：相关性热力图
    st.markdown("##### 15. 数值特征相关性热力图")
    numeric_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['PlayerID'], errors='ignore')
    corr = numeric_df.corr()
    fig15 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig15, use_container_width=True)