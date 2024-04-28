import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model has been placed on device: ", device)
class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
input_size = 64  # The size of the input tensor
hidden_size = 128  # The size of the hidden tensor
output_size = 64  # The size of the output tensor 
learning_rate = 0.001 # The rate in which the model will adjust
batch_size = 64 # The size of each game state
num_epochs = 10 # The evaluations

model = ChessNet(input_size, hidden_size, output_size) # Declare what the model
model.load_state_dict(torch.load('chess_net.pth'))
model.to(device) #Load our .pth model
model.eval()  # Set the model to evaluation mode

print(f"Model loaded from {'chess_net.pth'}") # model loaded!

def preprocess_input(chess_state):
    return torch.tensor(chess_state).float().flatten().to(device)  # Convert our chess state to a tensor
def decode_output(output): # Decode the output back into an original game state matrix
    return np.round(output.view(8, 8).detach().cpu().numpy()).astype(int) # Round each float returned by the model
def predict_next_state(model, input_state): # Pass the state into the model
    input_tensor = preprocess_input(input_state)
    with torch.no_grad():
        output = model(input_tensor)
    next_state = decode_output(output)
    return next_state # The final state

# Example of using the trained model to predict the next chess state
input_state = np.array([-4., -2., -3., -5., -6., -3., -2., -4.,
                        -1., -1., -1., -1., -1., -1., -1., -1.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         4.,  2.,  3.,  5.,  6.,  3.,  2.,  4.])

# Predict the next chess state
next_state = predict_next_state(model, input_state)
print("Predicted next chess state:")
print(next_state)
