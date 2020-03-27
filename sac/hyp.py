import torch

# Hyperparameters
ENV = 'HalfCheetah-v2'
TAU = 0.005
EPSILON = 1e-6
H_DIM = 256
LR = 3e-4
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 256
ALPHA = 0.01
GAMMA = 0.99 # 0.98
ENTROPY_TUNING = True # True
MAX_STEPS = 1000000
EXPLORATION_TIME = 10000
MIN_LOG = -20
MAX_LOG = 2
TENSORBOARD_LOGS = True
TENSORBOARD_LOGS_PATH = './tests/halfCheetahSAC'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')