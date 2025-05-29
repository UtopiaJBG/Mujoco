import mujoco
import mujoco.viewer
import time
import os

# --- Carregamento do modelo e dados ---
MODEL_XML_PATH = os.path.join("assets", "braco_ombro.xml")
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data  = mujoco.MjData(model)

# --- Parâmetros de simulação ---
simulation_wall_time_limit = 120  # segundos

# --- Exemplo de callback de controle (PD simples) ---
# Ajuste kp, kd e desired_qpos conforme seu robô
kp = 50.0
kd = 5.0
# vetor de posições alvo (mesmo tamanho de nq)
desired_qpos = [0.0] * model.nq  

def control_callback(model, data):
    # Para cada junta (assumindo uma-para-uma ctrl↔joint)
    for i in range(model.nq):
        # erro de posição e velocidade
        e_pos = desired_qpos[i] - data.qpos[i]
        e_vel =      0.0        - data.qvel[i]
        # lei de controle PD
        data.ctrl[i] = kp * e_pos + kd * e_vel
    # OBS: data.ctrl deve ter tamanho >= model.nu

# --- Lançando o managed viewer com callback de controle ---
with mujoco.viewer.launch(model, data, callback=control_callback) as viewer:
    print(f"Iniciando simulação por até {simulation_wall_time_limit} segundos...")
    start_time = time.time()

    # Enquanto a janela estiver aberta e não passar do limite de tempo
    while viewer.is_running() and (time.time() - start_time < simulation_wall_time_limit):
        # sincroniza render + física; o callback será chamado internamente
        viewer.sync()

    print("Simulação encerrada.")
