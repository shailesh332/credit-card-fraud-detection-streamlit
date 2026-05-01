import streamlit as st
import joblib
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Powered Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.big-title{
font-size:72px;
font-weight:800;
text-align:center;
color:#00f2ff;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#b0bec5;
margin-bottom:30px;
}

.kpi-card{
background:rgba(255,255,255,0.05);
padding:20px;
border-radius:15px;
text-align:center;
box-shadow:0px 6px 20px rgba(0,0,0,0.4);
}

.kpi-title{
font-size:16px;
color:#aaa;
}

.kpi-value{
font-size:28px;
font-weight:bold;
color:white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------


# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build correct model path
model_path = os.path.join(BASE_DIR, "models", "final_fraud_model.pkl")

# Load model
model = joblib.load(model_path)

threshold_path = os.path.join(BASE_DIR, "models", "best_threshold.txt")

with open("best_threshold.txt") as f:
    THRESHOLD = float(f.read())

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<p class="big-title">AI Powered Credit Card Fraud Detection Dashboard</p>', unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Real-time Machine Learning System for Detecting Fraudulent Credit Card Transactions</p>',
unsafe_allow_html=True
)

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
col1,col2,col3,col4 = st.columns(4)

col1.markdown("""
<div class="kpi-card">
<div class="kpi-title">MODEL</div>
<div class="kpi-value">Random Forest</div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="kpi-card">
<div class="kpi-title">FEATURES</div>
<div class="kpi-value">{len(feature_order)}</div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="kpi-card">
<div class="kpi-title">THRESHOLD</div>
<div class="kpi-value">{THRESHOLD}</div>
</div>
""", unsafe_allow_html=True)

col4.markdown("""
<div class="kpi-card">
<div class="kpi-title">SYSTEM STATUS</div>
<div class="kpi-value">ACTIVE</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# INPUT AREA
# --------------------------------------------------
tab1,tab2 = st.tabs(["Manual Transaction","Upload JSON"])

inputs = {}

with tab1:

    cols = st.columns(4)

    for i,f in enumerate(feature_order):
        with cols[i % 4]:
            inputs[f] = st.number_input(f,value=0.0)

    run = st.button("Analyze Transaction")

with tab2:

    uploaded = st.file_uploader("Upload Transaction JSON", type="json")

    if uploaded:
        inputs = json.load(uploaded)
        run = True
    else:
        run = False

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if run:

    df = pd.DataFrame([inputs])
    df = df.reindex(columns=feature_order, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    fraud = int(prob >= THRESHOLD)

    st.divider()

    col1,col2 = st.columns(2)

    # FRAUD GAUGE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text':"Fraud Risk Score"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"red"},
            'steps':[
                {'range':[0,40],'color':"green"},
                {'range':[40,70],'color':"orange"},
                {'range':[70,100],'color':"red"}
            ]
        }
    ))

    col1.plotly_chart(fig,use_container_width=True)

    # DONUT CHART
    fig2 = px.pie(
        names=["Normal","Fraud"],
        values=[1-prob,prob],
        color=["Normal","Fraud"],
        color_discrete_map={"Normal":"#2ecc71","Fraud":"#e74c3c"},
        hole=0.4
    )

    col2.plotly_chart(fig2,use_container_width=True)

    st.divider()

    # RISK LEVEL
    st.subheader("Fraud Risk Level")

    st.progress(prob)

    if prob > 0.7:
        st.error("🚨 High Fraud Risk")
    elif prob > 0.4:
        st.warning("⚠ Medium Risk")
    else:
        st.success("✅ Low Risk")

    st.divider()

    # PROBABILITY BAR
    st.subheader("Fraud vs Normal Probability")

    bar_df = pd.DataFrame({
        "Type":["Normal","Fraud"],
        "Probability":[1-prob,prob]
    })

    fig3 = px.bar(
        bar_df,
        x="Type",
        y="Probability",
        color="Type",
        color_discrete_map={"Normal":"#2ecc71","Fraud":"#e74c3c"},
        text="Probability"
    )

    fig3.update_layout(showlegend=False)

    st.plotly_chart(fig3,use_container_width=True)

    st.divider()

    # FEATURE IMPORTANCE
    st.subheader("Model Feature Importance")

    try:

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature":feature_order,
            "Importance":importance
        })

        imp_df = imp_df.sort_values("Importance",ascending=False).head(10)

        fig4 = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig4,use_container_width=True)

    except:
        st.info("Feature importance not available.")

    st.divider()

    # EXPLAINABLE AI
    st.subheader("Explainable AI – Feature Contribution")

    contributions = df.iloc[0].abs().sort_values(ascending=False).head(10)

    explain_df = pd.DataFrame({
        "Feature": contributions.index,
        "Contribution": contributions.values
    })

    fig_exp = px.bar(
        explain_df,
        x="Contribution",
        y="Feature",
        orientation="h",
        color="Contribution",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(fig_exp,use_container_width=True)

    st.divider()

    # TRANSACTION ANALYTICS
    st.subheader("Transaction Analytics")

    colA,colB = st.columns(2)

    with colA:

        fig_hist = px.histogram(
            df,
            x="Amount",
            nbins=10,
            title="Transaction Amount Distribution"
        )

        st.plotly_chart(fig_hist,use_container_width=True)

    with colB:

        scatter = px.scatter(
            df,
            x="Time" if "Time" in df.columns else df.columns[0],
            y="Amount",
            title="Transaction Pattern"
        )

        st.plotly_chart(scatter,use_container_width=True)

    st.divider()
    
    # DATA TABLE
    st.subheader("Transaction Data")

    st.dataframe(df)