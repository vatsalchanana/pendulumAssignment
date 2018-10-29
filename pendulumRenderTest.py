
# coding: utf-8

# # Model Based Learning

# Consider a pendulum with a some unknown mass and unknown dynamics as
# your system. Your system takes in as input a torque (clockwise or
# counter clockwise). Your goal is to compute a sequence of actions that ensure
# the pendulum can stay in an upright position. This is a classic model based
# problem and can be broken down to 2 pieces :
# 1. Learn the model of the system.
# 2. Using the learned model, use a planner to compute the sequence of states
# required to achieve the desired position.
# Part a) Implement a system that learns the model of the pendulum.
# The state of the pendulum at any time t is given as s(t) = [cos(θ), sin(theta), θ,  ̇θ].
# Your output action space is a continuous action space corresponding to the
# torque τ applied at the hinge τ ∈ (−2, 2). 

# 
# 
# # Part A) Implement a system modelling the pendulum

# In[2]:


import gym
import numpy as np
import math
env = gym.make('Pendulum-v0')
env.reset();


# ### Create training data

# In[3]:


def getThetaThetaDotFromData(observation):
    theta = math.atan2(observation[1], observation[0]);
    return [theta/math.pi, observation[2]]


# In[4]:


def getTrainingSample(observation, action):
    return np.array(getThetaThetaDotFromData(observation) + action.tolist());


# In[5]:


x = []
y = []
for i_episode in range(200):
    observation = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        old_observation = observation
        observation, reward, done, info = env.step(action)
        x.append(getTrainingSample(old_observation, action))
        y.append(np.array(getThetaThetaDotFromData(observation)))
x = np.array(x)
y = np.array(y)


# In[6]:


x


# In[7]:


assert(len(x) == len(y))


# ### Create a neural network

# In[8]:


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import os
import numpy as np
import tensorflow as tf
import h5py
import math
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.models import model_from_json

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# Create a 3 layer NN
# 
# Training set -  N X 3 dimensional inputs with features as $\theta, \theta^., \tau$  
# 
# Output: N X 2, the $\theta, \theta^.$ describing the next state

# In[9]:


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

# ### Get training plots

# In[15]:


# import matplotlib.pyplot as plt
# history.history['loss'] = history.history['loss']
# plt.plot(history.history['loss'])
# plt.title('Training loss vs Iterations')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training', 'Validation'], loc='upper left')
# plt.show()


# In[16]:


#observation = env.reset()
#action = env.action_space.sample()

#x_test = []
#x_test.append(np.array(getTrainingSample(observation,action)))
#x_test = np.array(x_test)
#x_test.shape


# In[17]:


#y_hat = model.predict(x_test)


# In[18]:


#observation, reward, done, info = env.step(action)
#(np.array(getThetaThetaDotFromData(observation)))


# In[19]:


#x_test, y_hat


# ## Part B

# Using the learned model from Part a) and implement a graph based
# planner that can compute the most optimal solution. The easiest graph based
# planner can be A∗ (reads as A-star). For example, A-star takes in as input the
# current state, evaluates reachable states from the current state by computing a
# heuristic and then picks the best possible next state according to this heuristic.

# In[262]:


torque_options = [float(x)/10 for x in range(-20, 21)]
print(torque_options);

# In[263]:


#stateMap = dict()
#stateCount = [0]
#decToKeep = 2


# In[264]:


def getStateIndex(state):
    v0  = round(state[0],decToKeep)
    v1  = round(state[1],decToKeep)
    roundedState = [v0, v1]

    tupleForState = (roundedState[0],roundedState[1]);
    if(tupleForState in stateMap):
        return stateMap[tupleForState]
    else:
        stateCount[0] =  stateCount[0] + 1
        stateMap[tupleForState] = stateCount[0]
        return stateCount[0]


# In[265]:


def checkExistence(a):
    roundedState = [round(a[0],decToKeep), round(a[1],decToKeep)]
    tupleForState = (roundedState[0],roundedState[1]);
    if(tupleForState in stateMap):
        return True
    else:
        return False


# In[266]:


def checkStateEquals(a,b):
    return getStateIndex(a) == getStateIndex(b)


# In[267]:


#Test
#state = (1,1.222222)
#print(getStateIndex(state))
#checkStateEquals((1,1.222),(1.0001,1.222222))
#checkExistence((1,1.222222))


# 
# A Star Algorithm :

# Defining a node to run A*
# 3 values at each node: 
# - parent 
# - values = $[\theta, \theta.]$
# - action

# In[268]:


class Node():

    def __init__(self, parent=None, values=None, action=None):
        self.parent = parent
        self.values = values
        self.action = action

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        v1 = self.values
        v2 = other.values
        return checkStateEquals((v1[0],v1[1]),(v2[0],v2[1]))
    
    def __str__(self):
        return str(self.values) + str(self.action)
    
    def __repr__(self):
        return str(self.values)


# In[269]:


import math
from queue import PriorityQueue

def getHeuristic(values, action):
    theta = values[0]
    thetaDot = values[1]
    return -(theta**2 + thetaDot**2 + 0.001 * action**2)


# In[313]:


def astarOptimized(start_val,end_val, torque_options, model, start_torque = 0):
    start_node = Node(None, start_val, 0)
    print("Starting:" + str(start_node))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end_val, 0)
    came_from = {}
    cost_so_far = {}
    completed = {}
    frontier = PriorityQueue()
    frontier.put((0,start_node))
    startNodeIndex = getStateIndex((start_node.values[0],start_node.values[1]))
    cost_so_far[startNodeIndex] = 0;
    came_from[startNodeIndex] = None
    completed[startNodeIndex] = False
    while not frontier.empty():
        current_node = frontier.get()[1]
        #print(current_node, stateCount)
        if(current_node == end_node):
            path = []
            current = current_node
            while current is not None:
                path.append(current.action)
                current = current.parent
            return(path[::-1])
            break
        
        current_node_idx = getStateIndex((current_node.values[0],current_node.values[1]))
        
        if(current_node_idx in completed and completed[current_node_idx]):
            continue
        for torque in torque_options:
            next_value = model.predict(np.array([current_node.values + list([torque])]))[0]
            if(~checkExistence((next_value[0], next_value[1]))):
                new_node = Node(current_node, list(next_value), torque)
                heuristic = getHeuristic(next_value, torque)
                nodeIndex = getStateIndex((new_node.values[0],new_node.values[1]))
                cost_so_far[nodeIndex] = cost_so_far[current_node_idx] + heuristic;
                came_from[nodeIndex] = (current_node_idx, torque)
                new_node.f = cost_so_far[nodeIndex]
                frontier.put((cost_so_far[nodeIndex],new_node))
                
        completed[current_node_idx] = True
        
    return []


# In[314]:


def getActionSequence(initialObservation):
    init_state = getThetaThetaDotFromData(initialObservation)
    final_state = [0,0]
    stateMap = dict()
    stateCount = [0]
    decToKeep = 1
    return astarOptimized(init_state, final_state, torque_options,model)


# In[320]:


a = env.reset();
print(a,getThetaThetaDotFromData(a))
stateMap = dict()
stateCount = [0]
decToKeep = 1
ans = getActionSequence(a)


# In[321]:


print(ans)

for i_episode in range(1):
    i = 0;
    for actionV in ans:
        env.render()
        action = [actionV]
        print(action)
        i+=1;
        observation, reward, done, info = env.step(action)
     
print ans