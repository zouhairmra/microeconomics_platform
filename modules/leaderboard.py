import streamlit as st
import pandas as pd
import os

def run():

    st.header("🏆 لوحة المتصدرين")

    file = "data/scores.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
        st.dataframe(df.sort_values("score", ascending=False))
    else:
        st.write("لا توجد نتائج بعد")
