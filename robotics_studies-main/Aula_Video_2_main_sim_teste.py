#!/usr/bin/env python3
import mujoco
import mujoco.viewer # Novo módulo para o visualizador
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Definições e Inicializações ---
SCRIPT_ABSOLUTE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_XML_PATH = os.path.join(SCRIPT_ABSOLUTE_DIR, "assets", "full_kuka_all_joints_gravity.xml")
print(f"Modelo XML: {MODEL_XML_PATH}")
# Carregar o modelo MuJoCo
try:
    print(f"Carregando modelo de: {MODEL_XML_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
except Exception as e:
    print(f"Erro ao carregar o modelo XML: {e}")
    exit(-1)

# Criar a estrutura de dados da simulação (MjData)
data = mujoco.MjData(model)

# Parâmetros da simulação e controle
# n_timesteps do script original: 10 segundos com timestep de 0.002 (do modelo)

n_timesteps = int(30 / model.opt.timestep) # isso retorna 30 sobre 0.002s por passo, ou seja, 15000 passos então dura 15 segundos

nq = model.nq       # Número de coordenadas generalizadas (posições)
nv = model.nv       # Número de graus de liberdade (velocidades)

# Arrays para log (usando nq para qpos)
qlog = np.zeros((n_timesteps, nq))

# --- Parâmetros do Controlador (do script original) ---
# Estado desejado
qpos_d = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
qvel_d = np.zeros(nv) # Original era np.zeros(7)
qacc_d = np.zeros(nv) # Original era np.zeros(7)

print(f"Informações do Modelo Carregado: nq (posições) = {nq}, nv (velocidades/DoF) = {nv}")

# Verificação de consistência para qpos_d (Kuka tem 7 DoF, nq e nv devem ser 7)
if model.nq != 7 or model.nv != 7:
    print(f"Alerta: O modelo carregado tem nq={model.nq}, nv={model.nv}. "
          f"A lógica de qpos_d e Kp/Kd original era para 7 DoF.")
if len(qpos_d) != model.nq: # qpos_d é definido mais abaixo, mas len(qpos_d) é 7
     print(f"Alerta Crítico: O qpos_d hardcoded tem 7 elementos, mas model.nq é {model.nq}. "
           "Isso causará problemas de dimensão.")
     # exit(-1) # Descomente para sair se houver incompatibilidade

# Ganhos do controlador (Kp, Kd) - Lógica do script original
Kp = np.eye(nv)
Kd = np.eye(nv)

# Aplicar lógica de ganhos original se nv for 7 (para Kuka)
if nv == 7:
    for i in range(nv):
        Kp[i, i] = 10 * (1 - 0.15 * i)
        Kd[i, i] = 1.2 * (Kp[i, i] / 2)**0.5
    Kd[6, 6] = Kd[6, 6] / 50
else:
    # Fallback para ganhos genéricos se nv não for 7, como um exemplo
    print(f"Aviso: Lógica de Kp/Kd original é para 7 DoF, mas o modelo atual tem nv={nv}. "
          "Usando ganhos de fallback genéricos.")
    default_kp_val = 100  # Exemplo de valor
    default_kd_val = 20   # Exemplo de valor
    for i in range(nv):
        Kp[i, i] = default_kp_val
        Kd[i, i] = default_kd_val

#A$
t = 0

# --- Loop de Simulação e Controle ---
print("Iniciando simulação... Pressione Ctrl+C no terminal para sair ou feche o visualizador.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    try:
        while t < n_timesteps and viewer.is_running():
            # --- Leitura de Dados ---
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            qlog[t, :] = qpos

            # --- Cálculo do Controle (Computed Torque Control) ---
            qpos_error = qpos_d - qpos
            qvel_error = qvel_d - qvel
            C_eq = data.qfrc_bias.copy()
            v_control = qacc_d + Kd @ qvel_error + Kp @ qpos_error

            # Matriz de massa/inércia M(q) ou H(q)
            # CORREÇÃO: data.qM está na forma empacotada (nM elementos), precisamos da densa (nv,nv)
            
            # 1. Inicializar a matriz densa H com zeros
            H_dense = np.zeros((model.nv, model.nv))  # Terá shape (7, 7) pois model.nv é 7
            
            # 2. Popular H_dense com a matriz de massa completa usando mj_fullM
            #    data.qM aqui é o vetor esparso/empacotado de nM elementos (28 para nv=7)
            mujoco.mj_fullM(model, H_dense, data.qM)
            
            # Agora H_dense é a matriz de massa (7,7) que podemos usar
            
            # --- Depuração Antes do Matmul (t={t}) ---
            print(f"--- Depuração Antes do Matmul (t={t}) ---")
            print(f"model.nv no loop: {model.nv}")
            print(f"Shape de data.qM (empacotado): {data.qM.shape}") # Deve ser (28,)
            print(f"Shape de H_dense (matriz de massa densa): {H_dense.shape}") # Deve ser (7,7)
            print(f"Shape de v_control: {v_control.shape}") # Deve ser (7,)
            
            # Lei de controle: u = M(q) * v_control + C_eq
            u = H_dense @ v_control + C_eq # Use H_dense aqui

            data.ctrl[:] = u
            mujoco.mj_step(model, data)
            viewer.sync()
            t += 1

    except KeyboardInterrupt:
        print("Simulação interrompida pelo usuário.")
    finally:
        if viewer and viewer.is_running():
            viewer.close()
        print("Visualizador fechado.")

# --- Plotar Resultados ---
if t > 0: # Apenas plota se a simulação rodou por alguns passos
    time_axis = np.arange(t) * model.opt.timestep # Eixo do tempo em segundos

    plt.figure(figsize=(12, min(8, 2 * nq))) # Ajusta altura da figura baseada no num de juntas
    plt.suptitle('Posição das Juntas ao Longo do Tempo (Novo `mujoco`)')

    for i in range(nq):
        plt.subplot(nq, 1, i + 1 if nq > 1 else 1) # Cria subplots para cada junta
        plt.plot(time_axis, qlog[:t, i], label=f'q{i} (real)')
        if i < len(qpos_d): # Verifica se qpos_d tem este componente
            plt.plot(time_axis, np.full(t, qpos_d[i]), 'k--', label=f'q{i} (desejado)')
        plt.ylabel(f'Posição q{i} (rad)')
        plt.legend(loc='best')
        if i == nq - 1 : # Adiciona label de tempo no último subplot
            plt.xlabel('Tempo (s)')
        else:
            plt.gca().axes.get_xaxis().set_ticklabels([]) # Remove ticks do eixo x para subplots intermediários

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta layout para o título principal e labels
    plt.show()

print("Simulação concluída.")