import streamlit as st

import modules.elasticity_game as elasticity_game
import modules.equilibrium_game as equilibrium_game
import modules.monopoly_game as monopoly_game
import modules.quiz_game as quiz_game
import modules.leaderboard as leaderboard

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

elif menu == "لوحة المتصدرين":
    leaderboard.run()
