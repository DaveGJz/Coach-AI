# app_llm.py
import streamlit as st
import time
import re
import openai
from data_model import (
    df_stats, model, label_encoders,
    predict_next_week_position, generate_analysis,
    plot_player_comparison, compute_player_stats, monte_carlo_ppr
)

# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(page_title="Fantasy PPR Advisor LLM", layout="wide")
st.title("🏈 Fantasy PPR Advisor – Conversacional")
st.write("""
Bienvenido al Fantasy PPR Advisor. 
Escribe tu pregunta sobre a quién iniciar la próxima semana, por ejemplo: `Hurts o Allen`.
La app te dará:
- Predicción PPR estimada
- Comparación gráfica
- Probabilidad de >20 PPR
- Recomendación explicada por IA
""")

# ---------------------------
# API Key OpenAI
# ---------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------------------
# Inicializar memoria de chat
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Funciones
# ---------------------------
def parse_question(question_text):
    parts = re.split(r'\s+[oO]\s+', question_text.strip())
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None

def get_players_filtered_last_position(df, player1_name, player2_name):
    df_p1 = df[df['player_display_name'].str.contains(player1_name, case=False, na=False)]
    df_p2 = df[df['player_display_name'].str.contains(player2_name, case=False, na=False)]
    if df_p1.empty or df_p2.empty:
        return None, None, None
    pos1 = df_p1.sort_values(by=['season','week'], ascending=[False,False]).iloc[0]['position']
    pos2 = df_p2.sort_values(by=['season','week'], ascending=[False,False]).iloc[0]['position']
    df_p1 = df_p1[df_p1['position'] == pos1]
    df_p2 = df_p2[df_p2['position'] == pos2]
    return df_p1, df_p2, (pos1, pos2)

def build_prompt(player1, player2, df_result):
    p1 = df_result[df_result['player_display_name'].str.contains(player1, case=False, na=False)].iloc[0]
    p2 = df_result[df_result['player_display_name'].str.contains(player2, case=False, na=False)].iloc[0]
    stats_p1 = compute_player_stats(df_stats, player1)
    stats_p2 = compute_player_stats(df_stats, player2)
    prompt = f"""
Tengo datos de fantasy PPR para dos jugadores de la NFL:

Jugador 1: {p1['player_display_name']}, Posición: {p1['position']}, Predicción PPR próxima semana: {p1['predicted_ppr']:.1f}, 
Promedio histórico: {stats_p1['avg_ppr']:.1f}, Desviación estándar: {stats_p1['std_ppr']:.1f}, Juegos: {stats_p1['games']}

Jugador 2: {p2['player_display_name']}, Posición: {p2['position']}, Predicción PPR próxima semana: {p2['predicted_ppr']:.1f}, 
Promedio histórico: {stats_p2['avg_ppr']:.1f}, Desviación estándar: {stats_p2['std_ppr']:.1f}, Juegos: {stats_p2['games']}
"""
    return prompt

def ask_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Mostrar historial previo
# ---------------------------
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    else:
        st.chat_message("assistant").write(chat["content"])

# ---------------------------
# Input de usuario siempre visible
# ---------------------------
user_message = st.chat_input("Escribe tu pregunta aquí...")

if user_message:
    st.session_state.chat_history.append({"role":"user","content":user_message})
    st.chat_message("user").write(user_message)

    # ---------------------------
    # Procesar la pregunta para extraer jugadores
    # ---------------------------
    player1, player2 = parse_question(user_message)
    if not player1 or not player2:
        answer = "No se pudieron identificar los jugadores en tu pregunta."
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role":"assistant","content":answer})
    else:
        df_p1, df_p2, (pos1, pos2) = get_players_filtered_last_position(df_stats, player1, player2)
        if df_p1 is None or df_p2 is None:
            answer = "No se encontraron ambos jugadores en los datos."
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append({"role":"assistant","content":answer})
        elif pos1 != pos2:
            answer = "Los jugadores no son de la misma posición. Comparación válida solo dentro de la misma posición."
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append({"role":"assistant","content":answer})
        else:
            position_input = pos1
            df_result, recommendation_ml = predict_next_week_position(
                df_stats, model, label_encoders, player1, player2, season=2025, position=position_input
            )
            if df_result.empty:
                st.chat_message("assistant").write(recommendation_ml)
                st.session_state.chat_history.append({"role":"assistant","content":recommendation_ml})
            else:
                # ---------------------------
                # Mostrar tabla
                # ---------------------------
                pos_map = {
                    "QB": ['player_display_name','team','position','predicted_ppr','passing_yards','passing_tds','rushing_yards','rushing_tds'],
                    "RB": ['player_display_name','team','position','predicted_ppr','rushing_yards','rushing_tds','receptions','receiving_yards','receiving_tds'],
                    "WR": ['player_display_name','team','position','predicted_ppr','receptions','receiving_yards','receiving_tds','rushing_yards','rushing_tds'],
                    "TE": ['player_display_name','team','position','predicted_ppr','receptions','receiving_yards','receiving_tds','rushing_yards','rushing_tds']
                }
                cols_to_show = pos_map.get(position_input, df_result.columns)
                st.subheader("Predicción PPR - Próxima semana")
                st.table(df_result[cols_to_show])

                # ---------------------------
                # Gráfica
                # ---------------------------
                st.subheader("Comparación gráfica")
                st.altair_chart(plot_player_comparison(df_result, position_input), use_container_width=True)

                # ---------------------------
                # Análisis breve
                # ---------------------------
                st.subheader("Análisis breve")
                st.write(generate_analysis(df_result, position_input=position_input))

                # ---------------------------
                # Monte Carlo
                # ---------------------------
                st.subheader("Probabilidad de >20 PPR")
                for player in [player1, player2]:
                    player_row = df_result[df_result['player_display_name'].str.contains(player, case=False, na=False)]
                    if not player_row.empty:
                        pred_ppr = player_row['predicted_ppr'].values[0]
                        prob = monte_carlo_ppr(player, df_stats, next_week_pred=pred_ppr)
                        st.markdown(f"**{player}**: Probabilidad de >20 PPR: {prob*100:.1f}%")

                # ---------------------------
                # LLM interactivo
                # ---------------------------
                st.subheader("Recomendación y explicación del LLM")
                prompt = build_prompt(player1, player2, df_result)
                prompt += f"\nPregunta del usuario: {user_message}\n"
                prompt += "Responde detalladamente incluyendo consistencia histórica, predicción PPR, matchup, riesgo y recomendación clara."
                explanation = ask_llm(prompt)
                # Mostrar progresivamente
                assistant_box = st.chat_message("assistant")
                placeholder = assistant_box.empty()

                typed = ""
                for ch in explanation:
                    typed += ch
                    placeholder.write(typed)
                    time.sleep(0.01) 
                st.session_state.chat_history.append({"role":"assistant","content":explanation})

# ---------------------------
# Tips de preguntas
# ---------------------------
st.info("💡 Ideas de preguntas:")
st.markdown("""
- ¿Quién tiene más probabilidad de pasar de 20 PPR?
- ¿Cómo influye la defensa contraria esta semana?
- Comparar consistencia histórica entre estos jugadores.
- ¿Qué jugador tiene mayor riesgo de fluctuación según estadísticas recientes?
""")
