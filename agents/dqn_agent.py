"""Models a single DQN agent making decisions for signals at one intersection.

Based on our problem representation, we want to have multiple agents, one for each intersection
in the network. Each of these agents are fed the information of the intersection they are being
trained on (local) and the information of their immediate neighbors (0,1,2,3,4 depending on 
network topology).

In this module, we initialize a single such agent. We define the config as input while instantiating
the agent object, which contains information about the state representation as well as hyperparameters.
More on this in the configs/ dir.

The basic functionality needed for this class is to instantiate the neural networks (target and online),
enable buffer replay to learn from experience, use epsilon greedy methods to select actions, apply
updates to the target network and other utilities like model saving and loading.
"""


class Agent:
    def __init__(self, agent_id:int, config:dict):
        self.agent_id:int = agent_id
        self.input_size:tuple = config.get('input_size')
        self.action_size:tuple = config.get('action_size')
        self.replay_buffer:list = []


    def _build_online_network(self):
        pass


    def _build_target_network(self):
        pass


    def _create_optimizer(self):
        pass


    def _create_loss_fn(self):
        pass


    def select_action(self, state):
        """Epsilon greedy explore-exploit tradeoff."""
        pass


    def store_experience(self, state, action, reward, next_state, done:bool):
        pass


    def learn(self):
        pass

    def update_target_network(self):
        pass

    def save_model(self, filepath:str):
        pass


    def load_model(self, filepath:str):
        pass