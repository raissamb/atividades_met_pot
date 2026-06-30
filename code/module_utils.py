# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:49:09 2024

@author: raissamb
"""

# %% Imports & Path Hardening
import sys
from pathlib import Path
import pandas as pd
import verde as vd
import numpy as np
#import json
import harmonica as hm
import matplotlib.pyplot as plt

# 1. Force resolve the absolute path to this utility script file first
CURRENT_FILE_PATH = Path(__file__).resolve()

# 2. Step back out of the modules/ directory to find your core 'code/' directory level
# File is in: /phd_programming/code/modules/module_utils.py
# .parent is: /phd_programming/code/modules/
# .parent.parent is the core 'code/' directory level
CODE_ROOT = CURRENT_FILE_PATH.parent

# 3. Derive the absolute repository workspace root directory location
PROJECT_HOME = CODE_ROOT.parent

# %% Static Roots Anchored Directly to Project Home Workspace
BASE_OUTPUT = PROJECT_HOME / "output"
DATA_ROOT = PROJECT_HOME / "data" / "synthetic_models_processed"

# %% General


def prepare_regular_grid(xmin: float, xmax: float,
                         ymin: float, ymax: float,
                         spacing: float, height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a structured regular grid using Verde.

    Parameters
    ----------
    xmin : Float
        Minimum value for x-axis (Easting following Verde coord. system).
    xmax : Float
        Maximum value for x-axis (Easting following Verde coord. system).
    ymin : Float
        Minimum value for y-axis (Northing following Verde coord. system).
    ymax : Float
        Maximum value for y-axis (Northing following Verde coord. system).
    spacing : Float
        Spacing value between coordinates.
    height : Float
        Height value that defines the planar surface of observation, Z positive upwards.

    Returns
    -------
    coords_grid : ndarray
        Tuple of grid coordinates.

    """
    # Explicitly unpack the 3 coordinates
    region = (xmin, xmax, ymin, ymax)
    easting, northing, upward = vd.grid_coordinates(
        region, spacing=spacing, extra_coords=height)
    # Return them as an explicit 3-element tuple
    return easting, northing, upward



def generate_geographic_grid(lat_origin, lon_origin, alt_origin, grid_max=100, step=5):
    """
    Gera um grid local (X, Y) de 0 a grid_max metros e converte para Lat/Lon
    baseado em uma coordenada de origem real.
    """
    # 1. Criar os vetores locais em metros (Easting e Northing)
    x_coords = np.arange(0, grid_max + step, step) # [0, 5, 10, ..., 100]
    y_coords = np.arange(0, grid_max + step, step)
    
    # Criar a malha regular (mesh)
    X, Y = np.meshgrid(x_coords, y_coords)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # 2. Fatores de conversão de metros para graus
    meters_per_degree_lat = 111132.0
    # Converte a latitude de origem para radianos para usar no cosseno
    lat_rad = np.radians(lat_origin)
    meters_per_degree_lon = 111132.0 * np.cos(lat_rad)
    
    # 3. Calcular Latitudes e Longitudes para cada ponto do grid
    latitudes = lat_origin + (Y_flat / meters_per_degree_lat)
    longitudes = lon_origin + (X_flat / meters_per_degree_lon)
    
    # Altitude constante para o grid (ou você pode somar uma topografia se quiser)
    altitudes = np.full_like(X_flat, alt_origin, dtype=float)
    
    # 4. Montar o DataFrame
    df_grid = pd.DataFrame({
        'Easting_X_m': X_flat,
        'Northing_Y_m': Y_flat,
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Altitude_m': altitudes
    })
    
    return df_grid


def convert_meters_to_lat_lon(lat_origin: float, 
                              lon_origin: float, 
                              alt_origin: float, 
                              coords_meters: tuple[np.ndarray, np.ndarray, np.ndarray],
                              ) -> pd.DataFrame:
    """
    Converte um grid regular (Easting, Northing, Upward) em coordenadas geográficas
    e calcula a altitude real (elipsoide/mar) necessária para o modelo IGRF.
    """
    x_meters = coords_meters[0].ravel()
    y_meters = coords_meters[1].ravel()
    z_meters = coords_meters[2].ravel() # Altura local do sensor (ex: 1m)
    
    # Fatores de conversão de metros para graus
    meters_per_degree_lat = 111132.0
    lat_rad = np.radians(lat_origin)
    meters_per_degree_lon = 111132.0 * np.cos(lat_rad)
    
    # Calcular Latitudes e Longitudes para cada ponto do grid
    latitudes = lat_origin + (y_meters / meters_per_degree_lat)
    longitudes = lon_origin + (x_meters / meters_per_degree_lon)
    
    # Altitude real combinada (Terreno + Altura do Sensor) para uso no IGRF
    altitudes_igrf = alt_origin + z_meters
    
    df_grid = pd.DataFrame({
        'Easting_X_m': x_meters,
        'Northing_Y_m': y_meters,
        'Height_local_m': z_meters,     # Mantém o Z local (1.0 m) claro para os alunos
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Altitude_IGRF_m': altitudes_igrf # O valor de ~61 m para o IGRF
    })
    
    return df_grid



def get_rotated_coordinates(angle: float, 
                            easting_local: np.ndarray, 
                            northing_local: np.ndarray, 
                            upward_local: np.ndarray):
    """
    Aplica a rotação de ângulo (em graus) nas coordenadas do sensor.
    """
    theta = np.radians(angle)
    X_rot = easting_local * np.cos(theta) - northing_local * np.sin(theta)
    Y_rot = easting_local * np.sin(theta) + northing_local * np.cos(theta)
    return (X_rot, Y_rot, upward_local)


def get_magnetization_vectors(props_list: list) -> np.ndarray:
    magnetization = []
    for sublist in props_list:
        intensity = sublist[0]
        inc = sublist[1]
        dec = sublist[2]
        mag_e, mag_n, mag_u = hm.magnetic_angles_to_vec(intensity, inc, dec)
        mag = (mag_e, mag_n, mag_u)
        magnetization.append(mag)
        
    # .T transposes the Nx3 array into a 3xN array
    return np.array(magnetization).T



def get_prism_tfa(coordinates: tuple[np.ndarray, np.ndarray, np.ndarray], 
                  dims: list[list], 
                  #magnetization_vec: np.ndarray, 
                  props: list[list],
                  incf: float, 
                  decf: float):
    
    # Get magnetization vector
    magnetization_vec = get_magnetization_vectors(props)
    
    # Get exact field components
    b_e, b_n, b_u = hm.prism_magnetic(coordinates, dims, magnetization_vec, field="b")
    b = (b_e, b_n, b_u)
    
    # Get tfa
    tfa_exact = hm.total_field_anomaly(b, incf, decf)
    
    return tfa_exact


def add_gaussian_noise(field: np.ndarray,
                       noise_percentage: float,
                       seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Function to create and add noise to a given field.

    Parameters
    ----------
    field : np.ndarray
        Input field.
    noise_percentage : float
        Noise percentage to setup noise amplitude in relation to maximum value of input field.
    seed : int
        Integer to start seed.

    Returns
    -------
    noisy_field : np.ndarray
        Field with noise.
    noise : np.ndarray
        Noise sequence.
    sigma : float
        Parameter to define noise scale.

    """
    rng = np.random.default_rng(seed)
    max_amplitude = np.max(np.abs(field))
    sigma = (noise_percentage / 100.0) * max_amplitude
    noise = rng.normal(loc=0.0, scale=sigma, size=field.shape)
    noisy_field = field + noise

    return noisy_field, noise, sigma



def plot_field_contourf(xcoords: np.ndarray, ycoords: np.ndarray,
                        field: np.ndarray,
                        title: str,
                        xlabel: str,
                        ylabel: str,
                        unit: str,
                        savepath: str | Path,
                        show: bool = False):
    """
    Function to plot

    Parameters
    ----------
    xcoords : 2d Array
        2D array containing the coordinates in the X axis, Cartesian System.
    ycoords : TYPE
        DESCRIPTION.
    field : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    unit : TYPE
        DESCRIPTION.
    figname : TYPE
        DESCRIPTION.
    mode : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    fig = plt.figure()
    plt.contourf(xcoords, ycoords, field)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().set_aspect("equal")
    plt.colorbar(label=unit)
    fig.savefig(Path(savepath), dpi=300, bbox_inches="tight")

    # 6. Handling Display vs. Memory
    if show:
        plt.show()
    else:
        # CRITICAL for loops: Close the figure to free memory!
        plt.close(fig)
    plt.show()
    plt.close()


def create_diurnal_variations(timing_min, amplitude=30, t_pico=720, seed=None):
    """
    Simula uma variação diurna típica (curva suave com pico próximo ao meio-dia).
    tempos_minutos: array com o tempo decorrido em minutos desde a meia-noite.
    amplitude: amplitude máxima da variação em nT (ex: 30 nT a 50 nT).
    t_pico: minuto do dia onde ocorre o pico (720 minutos = 12:00h).
    seed: semente inteira para garantir a reprodutibilidade do ruído.
    """
    # Converte o tempo para radianos considerando o ciclo de 24h (1440 minutos)
    # Centraliza o pico no t_pico
    fase = (timing_min - t_pico) * (2 * np.pi / 1440)
    
    # Onda suave (cosseno invertido/ajustado para ter pico em t_pico e mínimo à noite)
    diurna = amplitude * (np.cos(fase) + 1) / 2
    
    # Inicializa o gerador de números aleatórios recomendado do NumPy
    rng = np.random.default_rng(seed)
    
    # Adiciona um ruído instrumental/ambiental muito leve usando o gerador isolado
    ruido_base = rng.normal(0, 0.2, size=len(timing_min))
    
    return diurna + ruido_base

