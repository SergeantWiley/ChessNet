import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Hyperparameters:
    '''
        Defualt (can be modifed) hyper parameters

        Args:
            input_size (int): input tensor size
            hidden_size (int): hidden layer tensor size
            output_size (int): ouput tensor size
            learning_rate (float): rate in from Adam Optimizer
            batch_size (int): number of datasets per nueron
            num_epochs (int): number of epochs
        Returns:
            tuple: all hyperparameters
    '''
    def __init__(self,input_size=64,hidden_size=128,output_size=64
                 ,learning_rate=0.001,batch_size=64,num_epochs=50) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

class Preprocessing:
    '''
        Proccesses data from .pgn file to be interpreted later
        
        Args:
            filepath (str): Data orgin of the data to pre proccess
        Returns:
            np.array: all board states in a numpy matrix (array) format
    '''
    def __init__(self, file_path,gpu=False) -> None:
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

class MakeDataset:
    '''
        Converts board matrices to tensors

        Args:
            board_states (np.array): all states in the game collected as a numpy array
        Returns:
            torch.tensor: a pyTorch tensor array
    '''
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
    '''
        Create a Model
        
        Args:
            hyperparamters (tuple): hyperparameters created by the hyperparamaters class
        Returns
            torch.model (model): a ChessNet model
    '''
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
    
    '''
        Trains the actual model

        Args:
            dataset (tesnor): Dataset for training
            model (torch.model): a ChessNet Model
            hyperparameters (tuple): hyperparameters created bhy the hyperparameters class
            gpu (boolean): Whether to check and use the GPU (Cuda only)

    '''
    def __init__(self, dataset, model, hyperparameters,model_save_path='chess_net.pth', gpu=False) -> None:
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

import torch
import numpy as np

class Postprocessing:
    '''
        After training a model or saving it
        
        Args:
            model (torch.model): The architecture of the mode
            gpu (boolean): Whether to detect and use GPU (Cuda Only)
    '''
    def __init__(self, model, gpu=False):
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.model.eval()

    def preprocess_input(self, chess_state):
        return torch.tensor(chess_state).float().flatten().to(self.device)

    def decode_output(self, output):
        print(output)
        return np.round(output.view(8, 8).detach().cpu().numpy()).astype(int)

    def predict_next_state(self, input_state):
        input_tensor = self.preprocess_input(input_state)
        with torch.no_grad():
            output = self.model(input_tensor)
        next_state = self.decode_output(output)
        return next_state
    
    def load_model_data(self,model,model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()

#Example Usage

# Instantiate Preprocessing and prepare data
preprocess = Preprocessing('game.pgn')
board_states = preprocess.preprocess_data()

# Instantiate Hyperparameters
hyperparameters = Hyperparameters()

# Instantiate makeDataset and prepare dataset
training_data = MakeDataset(board_states)
dataset = training_data.prepare_dataset()

# Create a model using the instainited hyperparamters
make_model = Model(hyperparameters,gpu=True)
model = make_model.makeModel()

# Starting Training
train = Train(dataset, model, hyperparameters, gpu=True)
train.train()
#Save the model
model_path = 'chess_net.pth'
torch.save(model.state_dict(), model_path)
input_state = np.array([-4., -2., -3., -5., -6., -3., -2., -4.,
                        -1., -1., -1., -1., -1., -1., -1., -1.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                         4.,  2.,  3.,  5.,  6.,  3.,  2.,  4.])
post_train = Postprocessing(model,True)
model = post_train.load_model_data(model,model_path)
print(post_train.predict_next_state(input_state))