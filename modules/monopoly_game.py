import streamlit as st

def run():

    st.header("🏭 لعبة الاحتكار")

    price = st.slider("اختر السعر", 1.0, 100.0, 30.0)

    demand = 200 - 2 * price
    cost = 20
    profit = (price - cost) * demand

    st.metric("الربح", round(profit, 2))
