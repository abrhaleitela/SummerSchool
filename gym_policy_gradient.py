#
# gym_policy_gradient.py
# Implementing and testing policy gradients
# in basic control tasks
#
from argparse import ArgumentParser
import random
import numpy as np
from matplotlib import pyplot
import keras
from keras.layers import Dense, Flatten
import tensorflow as tf
import gym
from tqdm import tqdm
import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot
# This lower discount factor is good for the 
# simple Gym control tasks
GAMMA = 0.95
TRAINING_LOSS = []
class RandomAgent:
    """A random agent for discrete action spaces

    This works as a template for other agents.
    """
    def __init__(self, obs_shape, num_actions):
        """
        obs_shape: A tuple that represents observations shape, which
                   will be numpy arrays.
        num_actions: Int telling how many available actions there are
                     in the environment
        """
        self.num_actions = num_actions

    def step(self, obs):
        """Return an action for given observation

        obs is an observation of shape obs_shape (given
        in __init__)
        """
        return random.randint(0, self.num_actions - 1)

    def learn(self, trajectories):
        """Update agent policy based on trajectories

        Trajectories is a list of trajectories, each of which
        is its own list of [state, action, reward, done]
        """
        # Random agent does not learn
        #print("Derp-herp I am a random agent and I won't learn")
        return None

class PGAgentMC:
    """ Policy gradients with just the returns of whole episode """
    def __init__(self, obs_shape, num_actions):
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        # This is just an array of [0, 1, 2, ..., num_actions - 1]
        self.action_range = np.arange(num_actions)
        self.model = self._build_network(obs_shape, num_actions)
        self.update_function = self._build_update_operation(self.model, num_actions)
       
        
    def _build_network(self, input_shape, num_actions):
        """Build a Keras network for policy.

        Policy network maps inputs (observations) to probabilities
        of taking each action. Network is rather small and simple,
        just two Dense layers of 32 units.
        """
        #raise NotImplementedError("Implement small Keras model and then remove this line")
        model = keras.models.Sequential()
        model.add(Dense(32, activation= 'sigmoid', input_shape = input_shape))
        model.add(Dense(num_actions, activation="softmax"))
            # TODO
            # A small network: One layer with 32 units and "sigmoid" activation, 
            # followed by output layer with `num_actions` outputs and "softmax"
            # activation

        return model

    def _build_update_operation(self, model, num_actions):
        """Build policy gradient training operations for the model.

        Keras's standard `fit(x, y)` and `train_on_batch(x, y)` are not 
        quite suitable for policy gradient updates, so we will manually create
        update operations
        """

        # This delves into building graphs with Tensorflow: 
        # The following operations do not compute anything, they just
        # create the operations. The "Placeholders" will replaced
        # with proper values when we want to run computations on
        # the graph. (Sidenote: Newer Tensorflow versions use different 
        # type of API...)

        # The output tensor from model
        action_probabilities = model.output
        # We need couple of additional inputs for policy gradient: 
        #  1) Array of actions that were selected
        #  2) Returns observed (the "R" part of policy gradient)
        # Note that observations are already given to the model

        # Shape "None" is a wildcard: It can be of any shape.
        # I.e. the following will be 1D arrays of length N
        selected_action_placeholder = tf.placeholder(tf.int32, shape=(None,))
        return_placeholder = tf.placeholder(tf.float32, shape=(None,))

        # First, take the action probabilities of actions
        # we actually selected. 
        selected_actions = tf.stack(
            (tf.range(
                tf.shape(action_probabilities)[0], dtype=tf.int32), 
                selected_action_placeholder
            ),
            axis=1
        )

        selected_action_probabilities = tf.gather_nd(action_probabilities, selected_actions)

        # Remove this after you implement lines below
        

        #raise NotImplementedError("Implement the following parts in _build_update_operation and then remove this and above line")
        
        # TODO 
        # Note that you have to use tensorflow functions for following operations
        # - Take logarithm of the probabilities of select actions ("log_probs")
        log_probs = tf.math.log(selected_action_probabilities)
        # - Multiply returns and the log_probs together ("loss")
        loss =  return_placeholder * log_probs 
        # - Take mean over all elements in loss (It has losses from bunch of different samples)
        mean_loss = -tf.math.reduce_mean(loss)
        # - Create an optimizer with `tf.train.RMSPropOptimizer(1e-2, decay=0.0)`
        minimizer = tf.train.RMSPropOptimizer(1e-2, decay=0.0)
        # - Create a Tensorflow update operation to minimize the loss ("update_op")
        update_op = minimizer.minimize(mean_loss)
        # - Use keras.backend.function to create a function that takes in
        update_function = keras.backend.function([model.input, selected_action_placeholder, return_placeholder],[mean_loss],[update_op])
        #   all required inputs (placeholders and model.input), outputs loss and 
        #   updates the update_op
        # - Return the function created in previous step (instead of None)

        return update_function

    def step(self, obs):
        """Get action for the observation `obs` from the agent"""
        #raise NotImplementedError("Implement step function and then remove this line")
        p = self.model.predict(np.array(obs).reshape(1,4))[0]
        action = np.random.choice(np.arange(self.num_actions), 1, p = p )
        return action[0]
        # TODO
        # Get action from the agent. 
        # Our policy is now stochastic by its very nature:
        # The network (`self.model`) returns probabilities 
        # for each action, and we have to sample an 
        # action according to these probabilities.
        #  1. Get probabilities of the actions with `self.model.predict`
        #  2. Select an action according to these probabilities (randomly)
        #     (action is an integer between 0 and `self.num_actions - 1`).
        #  3. Return the action
        # Hint: `np.random.choice`
        

    def learn(self, trajectories):
        """ 
        `trajectories` is a list of trajectories, each of which is a list of
        experiences (state, action, reward, done) in the order they were 
        experienced in the game
        """
        #raise NotImplementedError("Implement learn function and remove this line")

        # TODO 
        # We need three elements to do policy gradient learning: 
        #   - Observations from the environment
        #   - The actions that were selected 
        #   - Returns for each state
        # Return for each state is the sum of discounted rewards, starting
        # from that state. E.g. For the first state in a trajectory it would be
        # 
        #return = 0
        #for i in range(length_of_the_trajectory):
        #   return += rewards[i] * GAMMA**i
        #
        observation_list = []
        action_list = []
        returns_list = [] 
        #print(trajectories[0])
        for episodic_trajectory in trajectories:
            for trajectory in episodic_trajectory:
                for i in range(len(trajectory)):
                    observation_list.append(trajectory[0])
                    action_list.append(trajectory[1])
                    returns_list.append(trajectory[2] * GAMMA**i)

        observation_list = np.array(observation_list)
        action_list = (np.array(action_list)-np.mean(np.array(action_list)))/np.std(np.array(action_list))
        returns_list = (np.array(returns_list)-np.mean(np.array(returns_list)))/np.std(np.array(returns_list))
        
        loss = self.update_function([observation_list, action_list, returns_list])
        TRAINING_LOSS.append(loss)
        # Proceeds as following:
        #   - Create three lists, one for each of these elements
        #   - Loop over the trajectories
        #     - For each experience in the trajectory, store the observation,
        #       the action and return from that state to their lists
        #   - Turn the lists into numpy arrays `np.array(list)`
        #   - Call the update function with 
        #     `self.update_function([observation_list, action_list, returns_list])
        #   - Above function returns the loss. Print it out so you can debug
        #     training.
        #raise NotImplementedError("Implement learn function and remove this line")
def play_game(env, agent):
    #raise NotImplementedError("Implement core step-loop here and then remove this line")
    terminal = False
    s1 = env.reset()
    env.render()
    trajectory = []
    while not terminal:
        action = agent.step(s1)
        s2, reward, terminal, info = env.step(action)
        trajectory_element = [s1, action, reward, terminal]
        trajectory.append(trajectory_element)
        s1 = s2
        #print(trajectory_element)
    
        
        
    # TODO
    # Implement loop that plays one game in env with agent, and then 
    # returns the trajectory.
    #   - Create empty list `trajectory`
    #   - Create the standard step-loop (while not done: ....)
    #   - For each observation, get action from agent with `agent.step(observation)`
    #   - Store experiences in the trajctory list.
    #       - One experience is list [observation, action, reward, done]
    #   - Return trajectory once game is over (done == True)
    return trajectory

def main(args):
    env = gym.make(args.env)

    # Assume box observations and discrete outputs
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Set your agent here
    agent = PGAgentMC(input_shape, num_actions)
    step_ctr = 0
    last_update_steps = 0
    trajectories = []
    y = []
    # The main training loop
    progressbar = tqdm(total=args.max_steps)
    while step_ctr < args.max_steps:
        # Play single game on environment and 
        # get the trajectory, update step_counter
        # and store trajectory.
        trajectory = play_game(env, agent)
        step_ctr += len(trajectory)
        trajectories.append(trajectory)
        episodic_reward = 0
        #print(trajectories[0][0][2])
        for one_trajectory in trajectory:
            episodic_reward += one_trajectory[2]
        y.append(episodic_reward)
    
        
        
        progressbar.update(len(trajectories))
        if step_ctr % args.nsteps == 0:
            agent.learn(trajectories)      
            trajectories = []
        # TODO 
        # Implement periodic training: 
        # Every args.nsteps steps we should call 
        # `agent.learn(trajectories)` and clear the list
        # of trajectories (we can not use the old trajectories after update)
    # TODO
    # Visualize episodic rewards with matplotlib here
    pyplot.plot(y)
    pyplot.xlabel("Game Numbers")
    pyplot.ylabel("Episodic_rewards")
    pyplot.show()
    #pyplot.plot(TRAINING_LOSS)
    #pyplot.xlabel("Game Numbers")
    #pyplot.ylabel("Loss")
    #pyplot.show()
    env.close()

if __name__ == '__main__':
    parser = ArgumentParser("Vanilla policy gradient on Gym control tasks")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max number of steps to play")
    parser.add_argument("--nsteps", type=int, default=500, help="Steps per update")
    args = parser.parse_args()
    main(args)
