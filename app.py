import streamlit as st

from modules.elasticity_game import run as elasticity_game
from modules.equilibrium_game import run as equilibrium_game
from modules.monopoly_game import run as monopoly_game
from modules.quiz_game import run as quiz_game
from modules.leaderboard import run as leaderboard

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
    elasticity_game()

elif menu == "لعبة التوازن":
    equilibrium_game()

elif menu == "لعبة الاحتكار":
    monopoly_game()

elif menu == "الاختبار":
    quiz_game()

elif menu == "لوحة المتصدرين":
    leaderboard()
