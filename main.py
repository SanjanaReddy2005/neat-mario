import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pickle
import os
import neat
import gym

import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
from gym.spaces import Box



env = gym_super_mario_bros.make('SuperMarioBros-v0')

env = JoypadSpace(env,[["right"], ["right", "A"]])


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=60)



env.reset()
new_state, reward, done, info = env.step(env.action_space.sample())


print(f"new_state:{new_state}\n Reward:{reward}\n Status:{done}\n info:{info}")
print(env.observation_space.shape)

generation = 38

def eval_genomes(genomes,config):
    global generation
    generation +=1

    for genome_id, genome in genomes:

        state = env.reset()
        reward = 0
        net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)

        done = False
        fitness = 0
        fitness_max =0
        frame =0
        count=0
        
        while not done:
            frame +=1
            state = torch.flatten(state)
            netout = net.activate(state)
            if(netout[0]>netout[1]):
                out = 0
            else:
                out = 1
            state, reward, done, info = env.step(out)
            fitness += reward
            genome.fitness = fitness
            if fitness > fitness_max:
                fitness_max = fitness
                count =0
            else:
                count +=1
            if count >100:
                done =True




            





def run(config):
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-571')
    # p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100))

    winner = p.run(eval_genomes,1000)

    with open("ex.pickle","wb") as f:
        pickle.dump(winner,f)
    






if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,'config.txt')

    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)

    run(config)


    

    