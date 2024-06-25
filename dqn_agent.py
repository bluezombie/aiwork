import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent
import random

class DQNAgent(Agent):
    def __init__(self, gym_env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device):
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, device)
        # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.policy_net = model.to(self.device)

        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)

        # Asignar un optimizador (Adam)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        
    
    def select_action(self, state, current_steps, train=True):
      
      self.epsilon = self.compute_epsilon(current_steps)
      print('Valor de Epsilon...')
      print(self.epsilon)

      # Implementar. Seleccionando acciones epsilongreedy-mente si estamos entranando y completamente greedy en otro caso.
      if train:
        # Entrenamos, por lo que las acciones deben ser epsilon-greedy
        if random.random() < self.epsilon:
          # Exploramos
          return self.env.action_space.sample()
        else:
          # Explotamos
          return self.policy_net(state).argmax().item()
      else:
         with torch.no_grad():
          # No entrenamos, por lo que las acciones deben ser completamente greedy
          return self.policy_net(state).argmax().item()
  

    def update_weights(self):
      if len(self.memory) > self.batch_size:
            # Resetear gradientes
            self.optimizer.zero_grad()

            # Obtener un minibatch de la memoria.
            mini_batch = self.memory.sample(self.batch_size)

            # Recordemos la estructura de las tuplas de la memoria.
            # Transition = namedtuple('Transition',('state', 'action', 'reward', 'done', 'next_state'))

            states = list(map(lambda x: x.state, mini_batch))
            actions = list(map(lambda x: x.action, mini_batch))
            rewards = list(map(lambda x: x.reward, mini_batch))
            dones = list(map(lambda x: x.done, mini_batch))
            next_states = list(map(lambda x: x.next_state, mini_batch))            

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            q_actual = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            max_q_next_state = self.policy_net(next_states).max(1)[0].detach()
            
            # Si alguno de los valores de max_q_next_state es cero, entonces el estado siguiente es terminal.
            # En este caso, el valor de max_q_next_state debería ser 0.

            for i in range(self.batch_size):
                if dones[i] == 0:
                    max_q_next_state[i] = self.policy_net(next_states[i]).max().detach()
          

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.    
            target = rewards + self.gamma * max_q_next_state * (1 - dones)

            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
            self.optimizer.zero_grad()
            self.loss_function(q_actual, target.unsqueeze(1)).backward()
            self.optimizer.step()
            