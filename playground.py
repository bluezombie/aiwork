import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))



memory = []

nueva_tupla = Transition(1, 2, 3, 4, 5)
memory.append(nueva_tupla)
segunda_tupla = Transition(11, 12, 13, 14, 15)
memory.append(segunda_tupla)
tercera_tupla = Transition(21, 22, 23, 24, 25)
memory.append(tercera_tupla)

tuples = random.sample(memory, 2)

states = list(map(lambda x: x.state, tuples))
actions = list(map(lambda x: x.action, tuples))

print('States')
print(states)
print('Actions')
print(actions)
