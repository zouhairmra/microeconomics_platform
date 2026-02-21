import streamlit as st
from modules import elasticity_game, equilibrium_game, quiz_game


st.set_page_config(page_title="Microeconomics Game Platform", layout="wide")

st.title("🎮 منصة ألعاب الاقتصاد الجزئي")

menu = st.sidebar.selectbox(
    "اختر اللعبة",
    [
        "لعبة المرونة",
        "لعبة التوازن",
        "لعبة الاحتكار",
        "الاختبار",
        "لوحة المتصدرين"
    ]
)

if menu == "لعبة المرونة":
    elasticity_game.run()

elif menu == "لعبة التوازن":
    equilibrium_game.run()

elif menu == "لعبة الاحتكار":
    monopoly_game.run()

elif menu == "الاختبار":
    quiz_game.run()


