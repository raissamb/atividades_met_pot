# %% Imports & Path Hardening
import sys
from pathlib import Path
import numpy as np
import harmonica as hm
import pandas as pd
import datetime
import ppigrf as ig

# 1. Force resolve the absolute path to this script file first
CURRENT_FILE_PATH = Path(__file__).resolve()
CODE_ROOT = CURRENT_FILE_PATH.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import module_utils as ut

# %% Universal Project Root Discovery
PROJECT_HOME = CODE_ROOT.parent
OUTPUT = PROJECT_HOME / "output"
figures_path = PROJECT_HOME / "output" / "figures"
figures_path.mkdir(exist_ok=True, parents=True)

# %% Global Configuration Setup
STUDENTS = ["aluno1","aluno2", "aluno3", "aluno4", "aluno5", "aluno6", "aluno7", "aluno8", "aluno9"]
NOISE_SEEDS = [23, 42, 67, 89, 2, 17, 45, 98, 7]
DV_SEEDS = [3, 78, 36, 11, 63, 5, 52, 90, 21]
NOISE_PERCENTAGE = 3

# Grid min and max points coordinates (meters)
xmin, xmax = 0, 100
ymin, ymax = 0, 100
spacing = 5
height = 1

# %% Models dictionary (Com a flag de Remanência integrada)
MODELS = {
    "model1": {
        "dims": [[25.0, 75.0, 25.0, 75.0, -20.0, -10.0]],
        "intensity": [3.0],
        "remanente": False, # Seguirá o campo regional do local
        "name": "model1",
    },
    "model2": {
        "dims": [[25.0, 75.0, 25.0, 75.0, -20.0, -10.0]],
        "intensity": [3.0],
        "remanente": True,  # Possui vetor de magnetização próprio fixo
        "inc_rem": [45.0],
        "dec_rem": [12.0],
        "name": "model2",
    },
    "model3": {
        "dims": [
            [25.0, 75.0, 55.0, 65.0, -20.0, -10.0],
            [50.0, 75.0, 80.0, 85.0, -14.0, -8.0],
        ],
        "intensity": [2.0, 2.0],
        "remanente": False, # Seguirá o campo regional do local
        "name": "model3",
    },
    "model4": {
        "dims": [
            [25.0, 75.0, 55.0, 65.0, -20.0, -10.0],
            [50.0, 75.0, 80.0, 85.0, -14.0, -8.0],
        ],
        "intensity": [2.0, 2.0],
        "remanente": True, # Direções de remanência independentes
        "inc_rem": [35.0, 25.0],
        "dec_rem": [25.0, 12.0],
        "name": "model4",
    },
}

# %% Locais de levantamento
LOCATIONS = {
    "local1": {
        "name": "Resolute_Polo",
        "lat": 74.69,
        "lon": -94.83, # Corrigido para negativo (West)
        "altitude": 60.0,
    },
    "local2": {
        "name": "Paris_MidLat",
        "lat": 48.85,
        "lon": 2.35,
        "altitude": 35.0,
    },
    "local3": {
        "name": "Macapa_Equador",
        "lat": 0.03,
        "lon": -51.06, # Corrigido para negativo (West)
        "altitude": 15.0,
    },
}

# %% Execução das simulações cruzadas (Locais x Modelos)

# 1. Criação do grid de observação local em metros via Verde
easting, northing, upward = ut.prepare_regular_grid(xmin, xmax, ymin, ymax, spacing, height)
coordinates = (easting, northing, upward)

# 2. Definição estrita da data da campanha (Uso do pd.Timestamp resolve o erro do ppigrf)
survey_date = pd.Timestamp("2026-06-25")

for local_key, local in LOCATIONS.items():
    print(f"\n>>> Executando simulações para: {local['name']} <<<")
    
    # Gerar coordenadas geográficas para o local atual
    df_survey = ut.convert_meters_to_lat_lon(local["lat"], local["lon"], local["altitude"], coordinates)
    
    # Criar vetor de tempo para o levantamento (8h às 19h)
    timestamps = pd.date_range(start="2026-06-25 08:00:00", periods=len(df_survey), freq="90s")
    df_survey['Time'] = timestamps
    minutos_desde_meia_noite = timestamps.hour * 60 + timestamps.minute + timestamps.second / 60
    
    # --- CÁLCULO PONTO A PONTO DO IGRF ( corrigido ) ---
    lat_points = df_survey['Latitude'].values
    lon_points = df_survey['Longitude'].values
    alt_points = df_survey['Altitude_IGRF_m'].values / np.array(1000.0)  # km
    
    # Passando o pd.Timestamp puro blinda contra o TypeError do ppigrf
    Be, Bn, Bu = ig.igrf(lon_points, lat_points, alt_points, survey_date)
    
    # Calcular a Intensidade Total do Campo Regional (F0)
    # Adicionamos .flatten() para converter de (1, 441) para (441,)
    F0_regional = np.sqrt(Be**2 + Bn**2 + Bu**2).flatten()
    #df_survey['F0_IGRF_nT'] = np.round(F0_regional, 2)
    
    # Fazer o mesmo achatamento para Be, Bn, Bu antes de calcular as médias angulares
    Be_flat = Be.flatten()
    Bn_flat = Bn.flatten()
    Bu_flat = Bu.flatten()
    
    # Calcular a Inclinação e Declinação MÉDIAS usando os vetores achatados
    incf_local = np.degrees(np.arctan2(Bu_flat, np.sqrt(Be_flat**2 + Bn_flat**2))).mean()
    decf_local = np.degrees(np.arctan2(Be_flat, Bn_flat)).mean()
    
    
    for model_key, model in MODELS.items():
        model_name = model["name"]
        dims = model["dims"]
        intensities = model["intensity"]
        
        # --- LÓGICA AUTOMÁTICA DE REMANÊNCIA ---
        props = []
        if model["remanente"]:
            for j in range(len(dims)):
                props.append([intensities[j], model["inc_rem"][j], model["dec_rem"][j]])
        else:
            for j in range(len(dims)):
                props.append([intensities[j], incf_local, decf_local])
        
        # Cálculo da Anomalia de Campo Total (TFA) via Harmonica
        tfa_exact = ut.get_prism_tfa(coordinates, dims, props, incf_local, decf_local)
        tfa_exact = tfa_exact.flatten()
        
        # --- LOOP DE ESTUDANTES ---
        for seed_noise, seed_dv, student in zip(NOISE_SEEDS, DV_SEEDS, STUDENTS):
            
            # 1. Adicionar ruído gaussiano ao sinal do modelo
            tfa_noise, _, _ = ut.add_gaussian_noise(tfa_exact, NOISE_PERCENTAGE, seed_noise)
            
            # 2. Variação diurna da Estação Base (Estática na origem do grid)
            variacao_diurna = ut.create_diurnal_variations(minutos_desde_meia_noite, amplitude=35, t_pico=750, seed=seed_dv)
            campo_total_estacao_base = F0_regional[0] + variacao_diurna
            
            # 3. Composição do dado coletado em campo (Ponto a Ponto)
            campo_observado_campo = F0_regional + tfa_noise + variacao_diurna
            
            # 4. DataFrame Completo (Versão do Orientador / Gabarito)
            df_gabarito = df_survey.copy()
            df_gabarito['F_observado_nT'] = np.round(campo_observado_campo, 2)
            df_gabarito['F0_IGRF_nT'] = np.round(F0_regional, 2) # Salvando o IGRF aqui!
            
            # 4b. DataFrame Filtrado (Versão do Aluno - Sem as respostas!)
            # Deixamos apenas o necessário para eles trabalharem
            colunas_aluno = ['Easting_X_m', 'Northing_Y_m', 'Height_local_m',
                             'Latitude', 'Longitude', 'Altitude_IGRF_m', 
                             'Time', 'F_observado_nT']
            df_aluno_campo = df_gabarito[colunas_aluno]
            
            df_aluno_base = pd.DataFrame({
                'Time': timestamps,
                'F_base_nT': np.round(campo_total_estacao_base, 2)
            })
            
            # 5. Exportação organizada com ID de Local, Modelo e Perfil
            student_path = PROJECT_HOME / "output" / student
            student_path.mkdir(exist_ok=True, parents=True)
            
            # Arquivos do Aluno (Prontos para distribuição)
            path_campo_aluno = student_path / f"levantamento_campo_{student}_{local_key}_{model_name}.csv"
            path_base_aluno = student_path / f"estacao_base_{student}_{local_key}_{model_name}.csv"
            
            # Arquivo Gabarito (Para você e seu Orientador usarem no Dashboard)
            path_gabarito = student_path / f"gabarito_{student}_{local_key}_{model_name}.csv"
            
            df_aluno_campo.to_csv(path_campo_aluno, index=False)
            df_aluno_base.to_csv(path_base_aluno, index=False)
            df_gabarito.to_csv(path_gabarito, index=False)

print("\n[SUCESSO] Todo o banco de dados sintéticos foi gerado com sucesso!")