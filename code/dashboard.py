import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuração da página do Streamlit
st.set_page_config(page_title="Geophysics Magnetic Survey Dashboard", layout="wide")

# Descobrir caminhos usando a lógica do seu projeto
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_HOME = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_HOME / "output"

# Reimportar as configurações do seu script original para manter a consistência
# (Certifique-se de que o mag_survey_creation.py esteja importável ou copie os dicts)
from mag_survey_creation import LOCATIONS, MODELS, STUDENTS

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.title("Configurações do Levantamento")

# 1. Seleção do Local
local_opcoes = {LOCATIONS[k]["name"]: k for k in LOCATIONS}
local_selecionado_name = st.sidebar.selectbox("Selecione o Local:", list(local_opcoes.keys()))
local_key = local_opcoes[local_selecionado_name]
local_dict = LOCATIONS[local_key]

# 2. Seleção do Modelo/Fonte
modelo_opcoes = {MODELS[k]["name"]: k for k in MODELS}
modelo_selecionado_name = st.sidebar.selectbox("Selecione o Modelo de Fonte:", list(modelo_opcoes.keys()))
model_key = modelo_opcoes[modelo_selecionado_name]
model_dict = MODELS[model_key]

# 3. Seleção do Aluno
aluno_selecionado = st.sidebar.selectbox("Inspecionar Ruído/Diurna do Aluno:", STUDENTS)

# --- CARREGAMENTO DOS DADOS ---
PASTA_ALUNO = OUTPUT_DIR / aluno_selecionado

# O Dashboard passa a ler o arquivo GABARITO que possui todas as colunas de validação
arquivo_campo = PASTA_ALUNO / f"gabarito_{aluno_selecionado}_{local_key}_{model_dict['name']}.csv"
arquivo_base = PASTA_ALUNO / f"estacao_base_{aluno_selecionado}_{local_key}_{model_dict['name']}.csv"

# --- INTERFACE PRINCIPAL ---
st.title("🧲 Dashboard de Levantamento Magnético Sintético")
st.markdown(f"**Análise de Dados para Atividade Didática** | Inspecionando: `{aluno_selecionado}`")

if not arquivo_campo.exists() or not arquivo_base.exists():
    st.error(f"⚠️ Os arquivos de dados para o modelo `{model_dict['name']}` no `{local_dict['name']}` ainda não foram gerados ou os nomes não coincidem. Rode o script principal primeiro!")
else:
    df_campo = pd.read_csv(arquivo_campo)
    df_base = pd.read_csv(arquivo_base)
    
    # Grid info
    n_pontos = len(df_campo)
    grid_side = int(np.sqrt(n_pontos))

    # --- PAINEL SUPERIOR: METADADOS E GEOMETRIA 3D ---
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📋 Informações da Estação e Campo Regional")
        
        # Tabela informativa compacta
        info_data = {
            "Parâmetro": ["Latitude", "Longitude", "Altitude Terreno", "Inclinação IGRF (Média)", "Declinação IGRF (Média)", "F0 Regional Médio"],
            "Valor": [
                f"{local_dict['lat']}° N", 
                f"{local_dict['lon']}°", 
                f"{local_dict['altitude']} m",
                f"{df_campo['F0_IGRF_nT'].mean():.2f} nT" if 'F0_IGRF_nT' in df_campo.columns else "N/A",
                f"{local_dict.get('incf', 'Calculado')}°",
                f"{local_dict.get('decf', 'Calculado')}°",
            ]
        }
        st.table(pd.DataFrame(info_data))
        
        st.markdown(f"**Tipo de Magnetização da Fonte:** {'Remanente Propriá' if model_dict['remanente'] else 'Induzida pelo Campo'}")

    with col2:
        st.subheader("📦 Geometria das Fontes Prismáticas (3D)")
        
        # Gráfico 3D da posição do prisma
        fig_3d = plt.figure(figsize=(5, 4))
        ax = fig_3d.add_subplot(111, projection='3d')
        
        # Desenhar o plano do sensor (Z = 1)
        xx, yy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
        ax.plot_surface(xx, yy, np.ones_like(xx) * df_campo['Height_local_m'].iloc[0], alpha=0.2, color='blue', label='Malha do Sensor')
        
        # Plotar os prismas do modelo selecionado
        for dim in model_dict["dims"]:
            w, e, s, n, bottom, top = dim
            # Vértices para desenhar o bloco 3D
            ax.bar3d(w, s, bottom, (e-w), (n-s), (top-bottom), color='red', alpha=0.6, edgecolor='black')
            
        ax.set_xlabel('Easting X (m)')
        ax.set_ylabel('Northing Y (m)')
        ax.set_zlabel('Z Upward (m)')
        ax.set_zlim(-25, 5)
        st.pyplot(fig_3d)

    st.markdown("---")

    # --- PAINEL INFERIOR: MAPAS E CURVAS TEMPORAIS ---
    st.subheader("🗺️ Mapas de Contorno e Sinais Observados")
    
    col_map1, col_map2, col_map3 = st.columns(3)
    
    # CORREÇÃO AQUI: Mudado de grid_size para grid_side obtido dinamicamente
    # Como o grid é quadrado (21x21 = 441), passamos a tupla (grid_side, grid_side)
    X = df_campo['Easting_X_m'].to_numpy().reshape((grid_side, grid_side))
    Y = df_campo['Northing_Y_m'].to_numpy().reshape((grid_side, grid_side))
    
    with col_map1:
        st.markdown("**Variação do Campo Regional (IGRF)**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        
        # Caso a coluna não exista por estar comentada no principal, recalculamos ou tratamos:
        if 'F0_IGRF_nT' in df_campo.columns:
            f0_data = df_campo['F0_IGRF_nT'].to_numpy().reshape((grid_side, grid_side))
        else:
            # Fallback caso a coluna tenha sido omitida no .csv
            f0_data = np.full_like(X, local_dict['altitude']) # apenas para não quebrar o plot
            
        cp = ax.contourf(X, Y, f0_data, cmap='viridis')
        fig.colorbar(cp, ax=ax, label='nT')
        ax.set_ylabel("Northing (m)")
        ax.set_xlabel("Easting (m)")
        st.pyplot(fig)

    with col_map2:
        st.markdown("**O que o Aluno Vê em Campo (Dado Bruto)**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        # Correção aqui também para a tupla (grid_side, grid_side)
        obs_data = df_campo['F_observado_nT'].to_numpy().reshape((grid_side, grid_side))
        cp = ax.contourf(X, Y, obs_data, cmap='bwr')
        fig.colorbar(cp, ax=ax, label='nT')
        ax.set_xlabel("Easting (m)")
        st.pyplot(fig)

    with col_map3:
        st.markdown("**Variação Diurna Gravada na Base (Tempo)**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        df_base['Time'] = pd.to_datetime(df_base['Time'])
        ax.plot(df_base['Time'], df_base['F_base_nT'], color='purple', lw=2)
        ax.set_xlabel("Hora do Dia")
        ax.set_ylabel("Campo Base (nT)")
        plt.xticks(rotation=45)
        st.pyplot(fig)