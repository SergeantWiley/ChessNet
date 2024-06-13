
# Chess Net

Chess Net is designed to teach machine learning to those unfamilier with complexities. Some prequsites are a prior understanding of overal python syntax. Specifically function and classes. Mathematics is not a requirement for understanding machine learning although is helpful for some more complex understanding




## Break down of the lessson

- libraries and setup
- Data Preproccessing
- Data Formatting
- Architecture
- Hyper paramters
- Training
- Loading Model
- Generate perdictions

## Libraries and setup

This is the required step and libraries for the code below to work
```python
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU exists
print("Training Device:", device)
```
## Data Preproccessing

First we have to see what the format of our raw data looks like.
This is 2 games in raw data format. 
```bash
1. e4 b5 2. g4 c5 3. Bxb5 a6 4. d4 axb5 5. Ne2 cxd4 6. Nxd4 Nc6 7. Nxb5 Qa5+ 8.
N5c3 Nd4 9. Be3 Nxc2+ 10. Qxc2 Qd8 11. Qd1 d5 12. Nxd5 e6 13. e5 exd5 14. Nc3
d4 15. Nb5 Ne7 16. Bxd4 Ng6 17. Nc3 Nxe5 18. Bxe5 f6 19. Bd4 f5 20. a4 fxg4 21.
b3 Ba3 22. Rc1 Bxc1 23. Qxc1 Bf5 24. Qd1 Bc2 25. Qxc2 g3 26. fxg3 h5 27. Qe4+
Kf8 28. Nb5 g5 29. Qf5+ Ke7 30. Qe5+ Kf7 31. Nd6+ Kg6 32. Nc4 h4 33. Qe6+ Kh5
34. Bb6 hxg3 35. Bxd8 Rhxd8 36. Qe7 Rd4 37. Qe5 g2 38. Qxd4 gxh1=Q+ 39. Ke2
Qxh2+ 40. Ke3 Rd8 41. Qxd8 Qa2 42. Qxg5+ Kxg5 43. Ne5 Qxb3+ 44. Kf2 Qxa4 45.
Nf7+ Kf4 46. Nd6 Qb4 47. Nc8 Qc5+ 0-1

1. e4 e5 2. Nf3 Nf6 3. Nc3 Bc5 4. d3 d6 5. Bg5 Nc6 6. Nxe5 Bd7 7. Nxd7 Ne5 8.
Bxf6 Qxd7 9. Bxg7 Rg8 10. Bf6 Nc6 11. Na4 Bb4+ 12. Nc3 Bc5 13. f4 Nd4 14. Bxd4
Rg6 15. Bxc5 dxc5 16. d4 cxd4 17. Qxd4 Qxd4 18. Ne2 O-O-O 19. Nxd4 Rgd6 20. Nf5
Rc6 21. Nd6+ Kb8 22. Nxf7 Rd4 23. Nd8 Rxd8 24. Rd1 Rxd1+ 25. Kxd1 Rd6+ 26. Kc1
Re6 27. e5 c5 28. f5 Rxe5 29. f6 Rf5 30. f7 Rxf7 31. Bc4 Rd7 32. g4 a6 33. g5
b5 34. Bxb5 axb5 35. Rd1 Rf7 36. Rd7 Rxd7 37. h4 Re7 38. h5 Re1+ 39. Kd2 Re5
40. g6 hxg6 41. hxg6 Re6 42. g7 Re8 43. b3 Rg8 44. c4 Rxg7 45. cxb5 Rg2+ 46.
Kc3 Rg3+ 0-1
```
Using the code below, we parse the data from our file and put all game states into the file and returns a list.

```python
def parse_pgn_moves(file_path):
    all_moves = []
    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            moves = [move.uci() for move in game.mainline_moves()]
            all_moves.extend(moves)
    return all_moves
```
All moves are now stored. Using the function below, we can pass a move into a python chess game engine

```python
def update_board(board, move):
    board.push(chess.Move.from_uci(move))
```
Before iterating through each move and game state, we need to convert board states to matrices in order to be transfered to tensors.

```Python
def board_to_matrix(board):
    matrix = np.zeros((8, 8), dtype=int)
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is not None:
                matrix[7 - rank][file] = piece.piece_type * (-1 if piece.color == chess.BLACK else 1)
    return matrix
```
Data Preproccessing setup completed. Now to run it. The varaibles we will need are
```python
file_path = 'game.pgn' #Source of the game file
moves = parse_pgn_moves(file_path) # Collect all moves
board = chess.Board() # Create a python-chess game engine
board_states = []
```
Then running a for loop to interate over moves, we update the board state and then return the board state and we append that to the board state tensor. 
```python
for move in moves:
    update_board(board, move)
    board_states.append(board_to_matrix(board))
```
Data Preproccessing Completed!
## Data Formatting 

Currently, our data is in a numpy matrix. Instead, it needs to be in a tensor. Every game state needs to be passed into the function below to format it for training later
```python
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
```
Next we set input_tensor and target_tensor. Our model will see each data point as two tensors. The input tensor and target_tensor will each have a board state. Since all board states are dependant on the last, we position one state then a state after it. This will come clear later when we get to the Architecture. 
``` python
dataset = prepare_dataset(board_states)
input_tensor, target_tensor = dataset[0]
```
Data Formatting Completed!

## Architecture

The Architecture only has one single class/function however it contains a lot of content
```python
class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
First is the fc1 and fc2. Each one of these are a layer of the network. The FC stands for Fully connected. In this case we only need 2 layers. The reason for this is the same reason why we did only 2 game states per data point. We want teach 2 concepts. One is the value of the piece and one is the value of the position. 

In chess, a queen near the middle of the board is more valuable then a pawn near the end of the board. The first layer is finding the value of the piece and then second is finding the value of the positions. 

Since each state is based off the state from the last, we move forward setting x to our first layer then after the value of the peice is determined or learned, it gets passed to find the value of the position that is being determined or learned so the new x is sent for evaluation which brings us to the next part, which are the hyper paramters

## Hyperparamaters

```python
input_size = 64  # Size of the input tensor
hidden_size = 128  # Size of the hidden layer
output_size = 64  # Size of the output tensor
learning_rate = 0.001
batch_size = 64
num_epochs = 50
```
A few things. Number one is the inptu_size. The input size is the same size as the tensor so we pass 64. The hidden_size for the model doesnt nessarily have to be the same size. The output_size similear to the input, must be the same size as the input tensor unless the output tensor is a different size then the output_size should be the same size as the tensor output size

The learning_rate is important since its tells how much each nuerons values can change per every epoch. Speaking of epochs, epochs are the number of testing and evaluation of the nueral networks performance. 

batch_size is the number of data points each nueron gets. A smaller number means more nuerons have to work togeather decreasing the chance of over fitting

With that, Hyperparamaters are completed!

## Training

Now for the fun part. Training begins by passing our dataset to the pytorch dataloader

```python
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```
Often its best to mix up different data sets to decrease overfitting even more and our dataloader does that for us. It also splits our data into testing and training sets for us as well unlike other non pytorch libraries.

Once we have loaded our data, we need to make a model

```python
model = ChessNet(input_size, hidden_size, output_size).to(device)
```
Our model is now loaded with the Architecture and dataset. However, a model needs a method of grading and improving its self so we pass what counts as a criteria for grading which is the Mean Squared Error. Then for our optimizer, we use the most common optimizer funciton, Adam.

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

Finally with the model ready for training, we intialize training with a standard training loop.

```python
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
```
Our loop will use the epochs we declared above and place everything on the device of our choice. This is optional and can be removed but its recommended as this is what allows the GPU to be used if there is one. 

We then get what the model is curerntly trained on then we calculate the loss

```python
# inside the for loop
outputs = model(inputs)
loss = criterion(outputs, targets)
```
These are just executing the varaibles and functions we delcared and described above

```python
# inside the for loop
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
So that we can monitor the progress, we include a standard print function. This is very commonly used to see the current status of the model

```python
print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

And of course, we want to save our model so we can use it later

```python
model_path = 'chess_net.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
```
Now our Model is trained and saved!

## Loading the model

This part is very similear to the Architecture and Hyperparamaters part
```python
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
```
A lot is the same besides the last 3. We pull our saved model and give it to our model variable. Its key we still declare the same Architecture and Hyperparamaters. The final part (model.eval()) puts the model into a type of state that allows us to pass unseen data we can then use its learning and apply that to a new game state

## Generating perdictions

First we have to do some Data Formatting but its not as much as before. Since we only have 1 game state and want to see the result is, we only need 3 small functions.

```python
def preprocess_input(chess_state):
    return torch.tensor(chess_state).float().flatten().to(device)

def decode_output(output): 
    return np.round(output.view(8, 8).detach().cpu().numpy()).astype(int)

def predict_next_state(model, input_state): the model
    input_tensor = preprocess_input(input_state)
    with torch.no_grad():
        output = model(input_tensor)
    next_state = decode_output(output)
    return next_state
```

These functions first take our chess state (our current game state) and proccess it into a form that the model would like. this is very similear to formatting the data we did before. 

Our chess_state/input_state will look like a numpy array

```python
# Example of using the trained model to predict the next chess state
input_state = np.array([-4., -2., -3., -5., -6., -3., -2., -4.,
                        -1., -1., -1., -1., -1., -1., -1., -1.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         4.,  2.,  3.,  5.,  6.,  3.,  2.,  4.])
```

For simplicity we harded coded a chess state. 
Negative values are black while positive are black pieces.
1s are pawns, 6s are kings, 5s are queens, 3s are bishops, 4s are rooks, and 2s are knights

Now all we have to do is run the predict_next_state function
```python
next_state = predict_next_state(model, input_state)
print("Predicted next chess state:")
print(next_state)
```
# ITS COMPLETE!
## We have our Chess Net!
But there is a problem. Its not very good and it doesnt understand whats a legal move and an illegal move. Thats fine since this was never intended to be a useful model. In matter a fact, its so bad, it makes up its own pieces but the goal was to learn and follow the proccess of how machine learning works and the proccess of preparing, formating, training, and implenting Machine learning models. 
