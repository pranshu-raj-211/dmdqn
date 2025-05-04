import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import random
from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter
from src.scripts.train import initialize_environment, SmoothedValue
from src.experimental.order_lanes import get_own_state
import wandb
import traci


EPISODES = 10
STEP_DURATION = 10.0
BATCH_SIZE = 128
SEED_VALUE = 1074
NUM_NODES = 9
NODE_FEATURES_SIZE = 9
NUM_QUEUES_IN_STATE = 4
SUMO_CFG_PATH = "src/sumo_files/scenarios/grid_3x3_lefthand/grid_3x3_10h.sumocfg"
SUMO_NET_PATH = "src/sumo_files/scenarios/grid_3x3_lefthand/grid_3x3_lht.net.xml"
ACTION_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
MAX_SIM_TIME = 2400


class TrafficGNN(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=32, out_channels=4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def get_junction_state(
    junction_id, structured_junction_lane_map, max_lanes_per_direction, current_sim_time
):
    state = get_own_state(
        junction_id,
        structured_junction_lane_map,
        max_lanes_per_direction,
        current_sim_time,
    )
    assert (
        len(state) == NODE_FEATURES_SIZE
    ), f"incorrect state for junction {junction_id}"
    return torch.tensor(state, dtype=torch.float32).clone().detach()


def get_graph_from_env(
    tl_junctions: list,
    structured_junction_lane_map,
    max_lanes_per_direction,
    current_sim_time,
):
    """Constructs graph from the environment."""
    x = torch.stack(
        [
            get_junction_state(
                junction_id=i,
                structured_junction_lane_map=structured_junction_lane_map,
                max_lanes_per_direction=max_lanes_per_direction,
                current_sim_time=current_sim_time,
            )
            for i in tl_junctions
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8],
            [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5],
        ],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index)


def get_reward(prev_state, next_state):
    prev_qs = prev_state.x[:, :NUM_QUEUES_IN_STATE].sum(dim=1)  # total queue per node
    next_qs = next_state.x[:, :NUM_QUEUES_IN_STATE].sum(dim=1)
    reward = (prev_qs - next_qs).sum().item()  # encourage reducing queue
    if (prev_qs == next_qs).all():
        reward += -40  # penalty for stagnation
    return reward


def select_actions(
    q_values,
    epsilon,
    global_step,
    exploration_only_steps=2_000,
    epsilon_min=0.01,
    epsilon_decay_steps=10_000,
):
    """
    Selects actions using the epsilon-greedy policy with epsilon decay.

    Args:
        q_values (torch.Tensor): Predicted Q-values for each action (shape: [num_nodes, num_actions]).
        epsilon (float): Current epsilon value for exploration.
        global_step (int): Current global step count.
        exploration_only_steps (int): Number of steps to explore before decaying epsilon.
        epsilon_min (float): Minimum value for epsilon.
        epsilon_decay_steps (int): Number of steps over which epsilon decays.

    Returns:
        torch.Tensor: Selected actions for each node (shape: [num_nodes]).
    """
    # Decay epsilon
    if global_step < exploration_only_steps:
        epsilon = 1.0
    elif epsilon > epsilon_min:
        epsilon = max(
            epsilon_min,
            1.0
            * torch.exp(
                torch.tensor(
                    -(global_step - exploration_only_steps) / epsilon_decay_steps
                )
            ).item(),
        )

    # Epsilon-greedy action selection
    if random.random() < epsilon:
        # Explore: Choose random actions
        return torch.randint(0, q_values.shape[1], (q_values.shape[0],)), epsilon
    else:
        # Exploit: Choose actions with the highest Q-values
        return q_values.argmax(dim=1), epsilon


def train_step(model, optimizer, transition, gamma=0.95):
    state, action, reward, next_state = transition
    model.train()
    pred = model(state.x, state.edge_index)  # (9, 4)
    target = model(next_state.x, next_state.edge_index).detach().max(dim=1)[0]  # (9,)

    chosen_q = pred[range(NUM_NODES), action]  # Q(s,a)
    expected_q = reward + gamma * target  # Bellman

    loss = F.mse_loss(chosen_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), pred.detach()


def main():
    wandb.init(
        project="gnn-traffic-rl",
        config={
            "episodes": EPISODES,
            "state_version": 3,
            "step_duration": STEP_DURATION,
        },
        settings=wandb.Settings(init_timeout=30, mode="offline"),
    )
    writer = SummaryWriter("runs/traffic_rl")
    smooth_reward = SmoothedValue(alpha=0.2)

    model = TrafficGNN(in_channels=9, hidden_channels=64, out_channels=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer()
    epsilon = 1.0

    tl_junctions, ordered_junction_lane_map = initialize_environment()
    tl_junctions.sort()

    for episode in range(EPISODES):
        traci.load(["-c", SUMO_CFG_PATH])
        global_state = {}
        current_time = traci.simulation.getTime()

        # Initialize global state
        for junction in tl_junctions:
            global_state[junction] = get_own_state(
                junction_id=junction,
                structured_junction_lane_map=ordered_junction_lane_map,
                max_lanes_per_direction=3,
                current_sim_time=current_time,
            )

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            actions = {}

            # Get actions for each junction
            for junction_id in tl_junctions:
                state = get_junction_state(
                    junction_id,
                    structured_junction_lane_map=ordered_junction_lane_map,
                    max_lanes_per_direction=3,
                    current_sim_time=current_time,
                )
                state_tensor = torch.tensor(state, dtype=torch.float32)
                # Construct the graph for the environment
                graph = get_graph_from_env(
                    tl_junctions=tl_junctions,
                    structured_junction_lane_map=ordered_junction_lane_map,
                    max_lanes_per_direction=3,
                    current_sim_time=current_time,
                )

                # Get the node features (state) and edge_index from the graph
                state_tensor = graph.x
                edge_index = graph.edge_index

                # Pass the state and edge_index to the model
                q_values = model(state_tensor, edge_index)
                action, epsilon = select_actions(q_values, epsilon, step_count)
                actions[junction_id] = action.item()
                action, epsilon = select_actions(q_values, epsilon, step_count)
                actions[junction_id] = action.item()

            # Apply actions and step simulation
            for junction_id, action in actions.items():
                traci.trafficlight.setPhase(junction_id, ACTION_MAP[action])

            target_time = current_time + STEP_DURATION
            while current_time < target_time:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                done = (
                    traci.simulation.getMinExpectedNumber() == 0
                    or current_time >= MAX_SIM_TIME
                )

            # Update state and calculate rewards
            next_global_state = {}
            rewards = {}
            for junction in tl_junctions:
                next_global_state[junction] = get_own_state(
                    junction_id=junction,
                    structured_junction_lane_map=ordered_junction_lane_map,
                    max_lanes_per_direction=3,
                    current_sim_time=current_time,
                )
                rewards[junction] = get_reward(
                    global_state[junction], next_global_state[junction]
                )

            # Store experiences in replay buffer
            for junction_id in tl_junctions:
                buffer.push(
                    global_state[junction_id],
                    actions[junction_id],
                    rewards[junction_id],
                    next_global_state[junction_id],
                )

            # Train model if buffer has enough samples
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                total_loss = 0
                for transition in batch:
                    loss, q_preds = train_step(model, optimizer, transition)
                    total_loss += loss

                avg_loss = total_loss / len(batch)

                wandb.log(
                    {
                        "reward": total_reward,
                        "loss": avg_loss,
                        "q_mean": q_preds.mean().item(),
                        "q_max": q_preds.max().item(),
                        "smoothed_reward": smooth_reward,
                    }
                )

            global_state = next_global_state
            step_count += 1

        torch.save(model.state_dict(), f"traffic_gnn_rl{episode}.pt")
        writer.close()


if __name__ == "__main__":
    random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    main()
