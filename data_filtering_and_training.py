import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU exists
print("Training Device:", device)
# Collect all moves from game.png
def parse_pgn_moves(file_path):
    all_games = []
    with open(file_path) as pgn_file:
        game_moves = []
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            moves = [str(move) for move in game.mainline_moves()]
            game_moves.extend(moves)
            # Check if the current game has ended (indicated by a move numbered "1.")
            if moves and moves[0].startswith("1."):
                all_games.append(game_moves)
                game_moves = []  # Reset moves for the next game
    return all_games



def update_board(board, move): # Pass each last state into the game engine
    board.push(chess.Move.from_uci(move))
    
def board_to_matrix(board): # Convert the state of the game into a matrix
    matrix = np.zeros((8, 8), dtype=int)
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is not None:
                matrix[7 - rank][file] = piece.piece_type * (-1 if piece.color == chess.BLACK else 1) #Range each peice by color. -1 for white and 1 for black
    return matrix

# Run the game extraction
file_path = 'game.pgn' #Source of the game file
moves = parse_pgn_moves(file_path)
board = chess.Board()
board_states = []
# Append every game to one varaible
for move in moves:
    update_board(board, move)
    board_states.append(board_to_matrix(board))
def board_to_tensor(board_state):
    return torch.tensor(board_state).float().flatten()

# Prepare dataset
def prepare_dataset(board_states):
    dataset = []
    for i in range(len(board_states) - 1):
        # Current board state
        input_board = board_states[i]
        # Next board state
        target_board = board_states[i + 1]
        # Convert board states to tensors
        input_tensor = board_to_tensor(input_board)
        target_tensor = board_to_tensor(target_board)
        # Add input and target tensors to the dataset
        dataset.append((input_tensor, target_tensor))
    return dataset


# Prepare dataset from board states
dataset = prepare_dataset(board_states)
print(dataset)
# Check the length of the dataset
print("Number of samples in the dataset:", len(dataset))
# Example of accessing a sample from the dataset
input_tensor, target_tensor = dataset[0]
print("Input tensor shape:", input_tensor.shape)
print("Target tensor shape:", target_tensor.shape)

# Define the neural network architecture
class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the hyperparameters
input_size = 64  # Size of the input tensor
hidden_size = 128  # Size of the hidden layer
output_size = 64  # Size of the output tensor
learning_rate = 0.001
batch_size = 64
num_epochs = 50

# Convert input and target tensors to DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model and move it to the device
model = ChessNet(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Move inputs and targets to the device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print loss after each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training finished!')
model_path = 'chess_net.pth'

# Save the model
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")