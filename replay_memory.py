import random
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class ReplayMemory:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.position = 0
        self.filled = False

    def add(self, state, action, reward, done, next_state):
      # Creamos la named tupla con los parámetros que recibimos
      nueva_tupla = Transition(state, action, reward, done, next_state)
      # Si llegamos al final de la memoria, volvemos al comienzo
      # seteando la posición en 0.
      if self.position == self.buffer_size:
        self.filled = True
        self.position = 0
      # Verificamos si la memoria se está escribiendo por primera vez
      if not self.filled:
         self.memory.append(nueva_tupla)
      else:
        # La memoria ya fue escrita con elementos
        # anteriormente.
        self.memory[self.position] = nueva_tupla
      
      self.position += 1

    def sample(self, batch_size):
      try:
        return random.sample(self.memory, batch_size)
      except ValueError:
        print("Se solicitó a la memoria más elementos de los que tiene almacenados.")

    def __len__(self):
      return len(self.memory)