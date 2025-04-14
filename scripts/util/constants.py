# scripts/util/constants.py
import torch

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debug flag
DEBUG = False

# Training hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
MAX_LENGTH = 128
ACCUMULATION_STEPS = 2
SAMPLE_SIZE = 0.1 if DEBUG else 1