import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
import pandas as pd
import os

def run():

    st.header("🎯 لعبة تعظيم الإيراد")

    name = st.text_input("اسم الطالب")

    slope = random.uniform(1,4)

    intercept = 200

    price = st.slider("اختر السعر", 1.0, 100.0, 20.0)

    quantity = intercept - slope * price

    revenue = price * quantity

    optimal_price = intercept/(2*slope)

    optimal_quantity = intercept - slope * optimal_price

    optimal_revenue = optimal_price * optimal_quantity

    st.metric("الإيراد الحالي", round(revenue,2))

    fig = go.Figure()

    P = np.linspace(1,100,100)
    Q = intercept - slope*P

    fig.add_trace(go.Scatter(x=Q,y=P,mode="lines"))

    fig.add_trace(go.Scatter(x=[quantity],y=[price],mode="markers"))

    st.plotly_chart(fig,use_container_width=True)

    if st.button("إنهاء الجولة"):

        score = round(revenue/optimal_revenue*100,1)

        st.success(f"نتيجتك: {score}/100")

        save_score(name,score,"elasticity")


def save_score(name,score,game):

    file="data/scores.csv"

    if os.path.exists(file):

        df=pd.read_csv(file)

    else:

        df=pd.DataFrame(columns=["name","score","game"])

    new=pd.DataFrame([[name,score,game]],columns=df.columns)

    df=pd.concat([df,new])

    df.to_csv(file,index=False)
