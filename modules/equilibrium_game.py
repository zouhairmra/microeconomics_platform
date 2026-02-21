import streamlit as st

def run():

    st.header("⚖ لعبة إيجاد التوازن")

    demand_slope = 2
    supply_slope = 1

    price = st.slider("اختر السعر", 1.0, 100.0, 20.0)

    Qd = 200 - demand_slope * price
    Qs = supply_slope * price

    st.metric("الطلب", round(Qd, 1))
    st.metric("العرض", round(Qs, 1))

    if abs(Qd - Qs) < 5:
        st.success("🎉 لقد وجدت سعر التوازن!")
