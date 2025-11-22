# data_model.py
import os
import pandas as pd
import numpy as np
import joblib
import nflreadpy as nfl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import altair as alt

# ---------------------------
# Configuración de columnas y archivos
# ---------------------------
offensive_features = [
    'week', 'season', 'position', 'position_group', 'team', 'opponent_team',
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions',
    'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost',
    'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost',
    'fumble_recovery_own', 'fumble_recovery_opp', 'special_teams_tds'
]

target_column = 'fantasy_points_ppr'
MODEL_FILE = "xgb_fantasy_ppr_model_2025.pkl"
ENCODER_FILE = "label_encoders.pkl"

# ---------------------------
# Cargar y preparar datos
# ---------------------------
def load_and_prepare_data(seasons=[2023, 2024, 2025]):
    df_stats = nfl.load_player_stats(seasons).to_pandas()
    
    # Si no existe fantasy_points_ppr, calcularlo aproximadamente
    if target_column not in df_stats.columns:
        df_stats[target_column] = (
            df_stats['passing_yards'] * 0.04 +
            df_stats['passing_tds'] * 4 -
            df_stats['passing_interceptions'] * 2 +
            df_stats['rushing_yards'] * 0.1 +
            df_stats['rushing_tds'] * 6 +
            df_stats['receptions'] * 1 +
            df_stats['receiving_yards'] * 0.1 +
            df_stats['receiving_tds'] * 6 -
            df_stats['rushing_fumbles_lost'] * 2 -
            df_stats['receiving_fumbles_lost'] * 2
        )
    return df_stats

# ---------------------------
# Entrenar o cargar modelo
# ---------------------------
def train_or_load_model(df):
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        model = joblib.load(MODEL_FILE)
        label_encoders = joblib.load(ENCODER_FILE)
        print("✅ Modelo y encoders cargados desde disco")
        return model, label_encoders

    df_model = df[offensive_features + [target_column]].copy()
    num_cols = df_model.select_dtypes(include='number').columns
    df_model[num_cols] = df_model[num_cols].fillna(0)

    cat_cols = ['position', 'position_group', 'team', 'opponent_team']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Modelo entrenado. RMSE: {rmse:.2f} puntos PPR")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(label_encoders, ENCODER_FILE)
    print("✅ Modelo y encoders guardados")

    return model, label_encoders

# ---------------------------
# Predicción próxima semana
# ---------------------------
def predict_next_week_position(df, model, label_encoders, player1_name, player2_name, season, position):
    df_hist = df[df['position'] == position]

    df_players = df_hist[df_hist['player_display_name'].str.contains(player1_name, case=False, na=False) |
                         df_hist['player_display_name'].str.contains(player2_name, case=False, na=False)]
    
    if df_players.empty:
        return pd.DataFrame(), f"No se encontraron jugadores {position} con esos nombres."
    
    last_week = df_players['week'].max()
    next_week = last_week + 1
    df_next = []

    for player in [player1_name, player2_name]:
        df_player = df_players[df_players['player_display_name'].str.contains(player, case=False, na=False)]
        last_stats = df_player.sort_values(by='week').iloc[-1].copy()
        last_stats['week'] = next_week
        df_next.append(last_stats)
    
    df_next = pd.DataFrame(df_next)
    
    # Transformar categóricas
    for col in label_encoders:
        le = label_encoders[col]
        df_next[col] = le.transform(df_next[col].astype(str))

    X_next = df_next[offensive_features]
    preds = model.predict(X_next)
    df_next['predicted_ppr'] = preds
    recommended_player = df_next.loc[df_next['predicted_ppr'].idxmax(), 'player_display_name']

    result = df_next[['player_display_name', 'team', 'position', 'predicted_ppr',
                      'passing_yards','passing_tds','rushing_yards','rushing_tds',
                      'receptions','receiving_yards','receiving_tds']].sort_values(by='predicted_ppr', ascending=False)
    return result.reset_index(drop=True), f"Recomendación: iniciar {recommended_player}"

# ---------------------------
# Análisis breve dinámico
# ---------------------------
def generate_analysis(df_result, position_input=None):
    if df_result.shape[0] < 2:
        return "No hay suficientes datos para generar análisis."
    
    player1 = df_result.iloc[0]
    player2 = df_result.iloc[1]
    
    pos_map = {
        "QB": ['predicted_ppr','passing_yards','passing_tds','rushing_yards','rushing_tds'],
        "RB": ['predicted_ppr','rushing_yards','rushing_tds','receptions','receiving_yards','receiving_tds'],
        "WR": ['predicted_ppr','receptions','receiving_yards','receiving_tds','rushing_yards','rushing_tds'],
        "TE": ['predicted_ppr','receptions','receiving_yards','receiving_tds','rushing_yards','rushing_tds']
    }
    if position_input:
        analysis_cols = pos_map.get(position_input, df_result.columns)
    else:
        analysis_cols = df_result.columns

    stats_p1 = ', '.join([f"{c}: {player1[c]}" for c in analysis_cols if c in df_result.columns])
    stats_p2 = ', '.join([f"{c}: {player2[c]}" for c in analysis_cols if c in df_result.columns])

    analysis = f"""
- {player1['player_display_name']} tiene PPR estimado de {player1['predicted_ppr']:.1f}. Destacan: {stats_p1}.
- {player2['player_display_name']} tiene PPR estimado de {player2['predicted_ppr']:.1f}. Destacan: {stats_p2}.
"""
    return analysis

# ---------------------------
# Gráfica compacta Altair
# ---------------------------
def plot_player_comparison(df_result, position_input):
    pos_map = {
        "QB": ['predicted_ppr','passing_yards','passing_tds','rushing_yards','rushing_tds'],
        "RB": ['predicted_ppr','rushing_yards','rushing_tds','receptions','receiving_yards','receiving_tds'],
        "WR": ['predicted_ppr','receptions','receiving_yards','receiving_tds','rushing_yards','rushing_tds'],
        "TE": ['predicted_ppr','receptions','receiving_yards','receiving_tds','rushing_yards','rushing_tds']
    }
    cols_to_plot = pos_map.get(position_input, df_result.columns)
    df_long = df_result.melt(
        id_vars=['player_display_name'],
        value_vars=cols_to_plot,
        var_name='estadística',
        value_name='valor'
    )
    chart = alt.Chart(df_long).mark_bar().encode(
        x='estadística:N',
        y='valor:Q',
        color='player_display_name:N',
        column='player_display_name:N'
    ).properties(width=80, height=300)
    return chart

# ---------------------------
# Estadísticas históricas del jugador
# ---------------------------
def compute_player_stats(df, player_name):
    df_player = df[df['player_display_name'].str.contains(player_name, case=False, na=False)]
    return {
        'avg_ppr': df_player['fantasy_points_ppr'].mean() if not df_player.empty else np.nan,
        'std_ppr': df_player['fantasy_points_ppr'].std() if not df_player.empty else np.nan,
        'games': df_player.shape[0]
    }

# ---------------------------
# Monte Carlo >20 PPR
# ---------------------------
def monte_carlo_ppr(player_name, df, next_week_pred, simulations=10000):
    df_player = df[df['player_display_name'].str.contains(player_name, case=False, na=False)]
    if df_player.empty:
        return np.nan
    avg = df_player['fantasy_points_ppr'].mean()
    std = df_player['fantasy_points_ppr'].std()
    if np.isnan(std) or std == 0:
        std = 1
    simulated = np.random.normal(loc=next_week_pred, scale=std, size=simulations)
    prob = np.mean(simulated > 20)
    return prob

# ---------------------------
# Inicializar datos y modelo
# ---------------------------
df_stats = load_and_prepare_data()
model, label_encoders = train_or_load_model(df_stats)
