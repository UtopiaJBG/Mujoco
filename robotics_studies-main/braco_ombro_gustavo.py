import mujoco
# import mujoco.viewer
import mujoco_viewer
import time
import os


MODEL_XML_PATH = os.path.join("assets", "braco_ombro.xml")
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
 
data = mujoco.MjData(model) 

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     simulation_wall_time_limit = 120
#     start_time_wall = time.time()

#     print(f"\nIniciando simulação por até {simulation_wall_time_limit} segundos...")

#     while viewer.is_running() and (time.time() - start_time_wall < simulation_wall_time_limit):
#         step_start_sim_time = time.time()

#         mujoco.mj_step(model, data)

#         viewer.sync()

#         time_until_next_step = model.opt.timestep - (time.time() - step_start_sim_time)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)
    
#     print("\nSimulação encerrada.")

viewer = mujoco_viewer.MujocoViewer(model, data)

simulation_wall_time_limit = 120
start_time_wall = time.time()

print(f"\nIniciando simulação com 'mujoco-python-viewer' por até {simulation_wall_time_limit} segundos...")
print("Pressione 'H' ou segure 'Alt' para ver os painéis de ajuda e informação.")

# --- MUDANÇA: O loop agora está no escopo principal e usa 'viewer.is_alive' ---
while viewer.is_alive and (time.time() - start_time_wall < simulation_wall_time_limit):
    step_start_sim_time = time.time()

    mujoco.mj_step(model, data)

    # --- MUDANÇA: 'viewer.sync()' é trocado por 'viewer.render()' ---
    viewer.render()

    # --- Lógica para manter o tempo real (permanece igual, é uma boa prática) ---
    time_until_next_step = model.opt.timestep - (time.time() - step_start_sim_time)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)

# --- MUDANÇA: Fecha o visualizador explicitamente no final ---
viewer.close()
print("\nSimulação encerrada.")