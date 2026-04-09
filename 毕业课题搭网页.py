import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型和标准化器
model = joblib.load('model_lgbm.pkl')
scaler = joblib.load('scaler.pkl')

continuous_features = [
    "PASP",           # 肺动脉收缩压
    "LVEDD",          # 左心室舒张末期内径
    'CPB_duration',   # 体外循环时间
    'Intro_qjssxs.'   #去甲肾上腺素
]

asa_options = {1: 'I级', 2: 'II级', 3: 'III级', 4: 'IV级', 5: 'V级'}
device_options = {0: '未使用', 1: '使用'}

# 注意：顺序必须与模型训练时完全一致
all_feature_names = [
    "PASP", "LVEDD", 'CPB_duration', 'Intro_qjssxs.',   # 4个连续变量
    "ASA",                                              # ASA分级（数值）
    'LVAD_IABP'                                         # 心脏辅助装置（0/1）
]


# ========== 3. 页面标题 ==========
st.title("心脏术后急性肾损伤风险预测系统")
#st.markdown("基于LightGBM机器学习模型，预测心脏术后AKI的严重程度分级")

# ========== 4. 创建输入区（两栏布局） ==========
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("请输入患者临床特征")
    # 连续变量输入
    PASP = st.number_input("肺动脉收缩压（PASP，mmHg）:", min_value=10.0, max_value=120.0, value=40.0, step=1.0, help="术前心脏超声测得的肺动脉收缩压")
    LVEDD = st.number_input("左心室舒张末期内径（LVEDD，mm）:", min_value=30.0, max_value=90.0, value=50.0, step=1.0, help="术前心脏超声测得的左心室舒张末期内径")
    CPB_duration = st.number_input("体外循环时间（CPB时间，min）:", min_value=0, max_value=300, value=90, step=5, help="术中体外循环持续时间")
    Intro_qjssxs = st.number_input("去甲肾上腺素剂量（μg/kg/min）:", min_value=0.0, max_value=2.0, value=0.10, step=0.01, format="%.2f", help="术中持续输注的去甲肾上腺素剂量")
    # 分类变量输入
    ASA = st.selectbox("ASA分级:", options=list(asa_options.keys()), format_func=lambda x: asa_options[x], help="美国麻醉医师协会体格状态分级")
    LVAD_IABP = st.selectbox("术中是否使用心脏辅助装置:", options=list(device_options.keys()), format_func=lambda x: device_options[x], help="术中是否使用主动脉内球囊反搏（IABP）或左心室辅助装置（LVAD）")
    predict_clicked = st.button("开始预测", type="primary", use_container_width=True)

with right_col:
    # 右侧结果区（占位符）
    result_placeholder = st.empty()

# ========== 5. 预测逻辑 ==========
if predict_clicked:
    # 5.1 收集输入数据
    input_data = {
        "PASP": PASP,
        "LVEDD": LVEDD,
        "CPB_duration": CPB_duration,
        "Intro_qjssxs.": Intro_qjssxs,
        "ASA": ASA,
        "LVAD_IABP": LVAD_IABP
    }
    
    input_df = pd.DataFrame([input_data])
    
    # 5.2 对连续变量进行标准化
    input_df[continuous_features] = scaler.transform(input_df[continuous_features])
    
    # 5.3 按模型训练时的顺序提取特征
    final_input = input_df[all_feature_names].values
    
    # 5.4 预测概率（三分类）
    predicted_proba = model.predict_proba(final_input)[0]
    # 假设顺序：0=无AKI，1=AKI 1期，2=AKI 2/3期
    prob_no_aki = predicted_proba[0]
    prob_stage1 = predicted_proba[1]
    prob_stage23 = predicted_proba[2]
    
    # 5.5 确定预测类别
    predicted_class = np.argmax(predicted_proba)
    class_names = {0: "无AKI", 1: "AKI 1期（轻度）", 2: "AKI 2/3期（中重度）"}
    predicted_label = class_names[predicted_class]
    
    # 5.6 风险分层逻辑（根据中重度AKI概率）
    if prob_stage23 >= 0.5:
        risk_level = "高风险"
        risk_color = "error"
    elif prob_stage23 >= 0.3:
        risk_level = "中风险"
        risk_color = "warning"
    else:
        risk_level = "低风险"
        risk_color = "success"
    
    # 5.7 显示结果
    with result_placeholder.container():
        st.subheader("预测结果")

        # 显示详细概率
        st.markdown("**各类别概率：**")
        col1, col2, col3 = st.columns(3)
        col1.metric("无AKI", f"{prob_no_aki:.1%}")
        col2.metric("AKI 1期（轻度）", f"{prob_stage1:.1%}")
        col3.metric("AKI 2/3期（中重度）", f"{prob_stage23:.1%}")
        
        # 显示预测类别
        if predicted_class == 0:
            st.success(f"**预测结果：{predicted_label}**")
        elif predicted_class == 1:
            st.info(f"**预测结果：{predicted_label}**")
        else:
            st.error(f"**预测结果：{predicted_label}**")
        
        st.caption("注：本预测结果仅供临床参考，具体诊疗决策请结合患者实际情况由专业医师判断。")
# ========== 6. 页面底部说明 ==========
st.markdown("---")
st.caption("模型基于LightGBM算法构建，预测特征包括：PASP、LVEDD、去甲肾上腺素剂量、CPB时间、ASA分级及心脏辅助装置使用。")
