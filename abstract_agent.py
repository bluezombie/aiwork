import torch
import torch.nn as nn
from replay_memory import ReplayMemory, Transition
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import show_video
import numpy as np

class Agent(ABC):
    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device=None):
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Funcion phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente 
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_i
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block

        self.total_steps = 0
    
    def train(self, number_episodes = 50000, max_steps_episode = 10000, max_steps=1000000, writer_name="default_writer_name"):
      # Mantenemos el listado de recompensas.
      rewards = []
      #To-Do: Verificar si la llamada a total_steps es correcta
      total_steps = 0
      writer = SummaryWriter(comment="-" + writer_name)

      for ep in tqdm(range(number_episodes), unit=' episodes'):
        if total_steps > max_steps:
            break
        
        # Observar estado inicial como indica el algoritmo
        current_episode_reward = 0.0
        
        # Reseteamos el ambiente
        state = self.env.reset()

        for s in range(max_steps):

            # Seleccionar accion usando una política epsilon-greedy.
            action = self.select_action(state, total_steps, train=True)

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            newobs, reward, done, truncated, _ = self.env.step(action)

            current_episode_reward += reward
            total_steps += 1

            # Guardar la transicion en la memoria
            # Recordemos la estructura de la replay memory
            #    def add(self, state, action, reward, done, next_state):
            self.memory.add(state, action, reward, done, newobs)

            # Actualizar el estado
            state = newobs

            # Actualizar el modelo
            self.update_weights()


            if done or truncated: 
                break
        
        rewards.append(current_episode_reward)
        mean_reward = np.mean(rewards[-100:])
        writer.add_scalar("epsilon", self.epsilon, total_steps)
        writer.add_scalar("reward_100", mean_reward, total_steps)
        writer.add_scalar("reward", current_episode_reward, total_steps)

        # Report on the traning rewards every EPISODE BLOCK episodes
        if ep % self.episode_block == 0:
          print(f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}")

      print(f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}")

      torch.save(self.policy_net.state_dict(), "GenericDQNAgent.dat")
      writer.close()

      return rewards
    
        
    def compute_epsilon(self, steps_so_far):

        # Descenso lineal de epsilon.
        # self.epsilon =  self.epsilon_f + (self.epsilon_i - self.epsilon_f) * \
        #       (1 - min(1.0, steps_so_far / self.epsilon_anneal_time))

        return  self.epsilon_f + (self.epsilon_i - self.epsilon_f) * \
               (1 - min(1.0, steps_so_far / 0.01))


    
    def record_test_episode(self, env):
        done = False
    
        # Observar estado inicial como indica el algoritmo 
        
        env.start_video_recorder()
        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.
            action = self.select_action(self.env, self.total_steps, False)

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.

            if done:
                break      

            # Actualizar el estado  
        env.close_video_recorder()
        env.close()
        show_video()

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        pass

    @abstractmethod
    def update_weights(self):
        pass