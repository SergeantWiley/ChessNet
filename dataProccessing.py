import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from chessNet import ChessNet
from chessNet import Hyperparameters
class Preprocessing:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def parse_pgn_moves(self):
        all_moves = []
        with open(self.file_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                moves = [move.uci() for move in game.mainline_moves()]
                all_moves.extend(moves)
        return all_moves
    
    def update_board(self, board, move): # Pass each last state into the game engine
        board.push(chess.Move.from_uci(move))
        
    def board_to_matrix(self, board): # Convert the state of the game into a matrix
        matrix = np.zeros((8, 8), dtype=int)
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece is not None:
                    matrix[7 - rank][file] = piece.piece_type * (-1 if piece.color == chess.BLACK else 1) #Range each peice by color. -1 for white and 1 for black
        return matrix
    
    def preprocess_data(self):
        # Run the game extraction
        moves = self.parse_pgn_moves()
        board = chess.Board()
        board_states = []
        # Append every game to one varaible
        for move in moves:
            # Try to append append the new game state to the board engine. If it doesnt work, empty board
            try:
                self.update_board(board, move)
                board_states.append(self.board_to_matrix(board))
            except:
                board.empty()
        return board_states
class makeDataset:
    
    def __init__(self,board_states) -> None:
        self.board_states = board_states

    def board_to_tensor(self,board):
        return torch.tensor(board).float().flatten()
    # Prepare dataset
    def prepare_dataset(self):
        dataset = []
        for i in range(len(self.board_states) - 1):
            # Current board state
            input_board = self.board_states[i]
            # Next board state
            target_board = self.board_states[i + 1]
            # Convert board states to tensors
            input_tensor = self.board_to_tensor(input_board)
            target_tensor = self.board_to_tensor(target_board)
            # Add input and target tensors to the dataset
            dataset.append((input_tensor, target_tensor))
        return dataset

class Model:
    def __init__(self,hyperparameters,gpu=False) -> None:
        self.hyperparameters = hyperparameters
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"

    def makeModel(self):
        input_size = self.hyperparameters.input_size
        hidden_size = self.hyperparameters.hidden_size
        output_size = self.hyperparameters.output_size
        model = ChessNet(input_size, hidden_size, output_size).to(self.device)
        return model

class Train:
    def __init__(self, dataset, model, hyperparameters, gpu=False) -> None:
        self.dataset = dataset
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.hyperparameters = hyperparameters
        self.model = model
    def data_loading(self):
        # Destructure hyperparameters
        batch_size = self.hyperparameters.batch_size
        input_size = self.hyperparameters.input_size
        hidden_size = self.hyperparameters.hidden_size
        output_size = self.hyperparameters.output_size
        learning_rate = self.hyperparameters.learning_rate
        
        # Convert input and target tensors to DataLoader
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        # Initialize the model and move it to the device

        self.model = ChessNet(input_size, hidden_size, output_size).to(self.device)
        # Define loss function and optimizer

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        return train_loader,criterion,optimizer
    
    def train(self):
        num_epochs = self.hyperparameters.num_epochs
        train_loader,criterion,optimizer = Train.data_loading(self)
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                # Move inputs and targets to the device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            # Print loss after each epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    

# Instantiate Preprocessing and prepare data
preprocess = Preprocessing('game.pgn')
board_states = preprocess.preprocess_data()

# Instantiate Hyperparameters
hyperparameters = Hyperparameters()

# Instantiate makeDataset and prepare dataset
training_data = makeDataset(board_states)
dataset = training_data.prepare_dataset()

# Instantiate model and move it to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = ChessNet(hyperparameters.input_size, hyperparameters.hidden_size, hyperparameters.output_size)
make_model = Model(hyperparameters,gpu=True)
model = make_model.makeModel()
# Starting Training
train = Train(dataset, model, hyperparameters, gpu=True)
train.train()