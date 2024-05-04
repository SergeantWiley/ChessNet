
## Chess Net - Usage

- Install
```bash
git clone https://github.com/SergeantWiley/ChessNet.git
```
- Install Requirments

```bash
pip install -r requirements.txt
```

## Summary of classes
    - ChessNet - A basic Linear Architecture FeedForward Nueral Network
    - Hyperparameters - The settings the Architecture will Usage
    - Preprocessing - Used to oragnize and prepare data for training
    - makeDataset - Converts the Preprocessing Data to a dataset that can be used during training
    - Model- Makes the actual model based off the ChessNet and Hyperparameters class
    - Train - Trains the model to learn Chess
    - Postprocessing - Used to save, load, and use the model for perdictions. Can also be used to load pretrain model into an empty model
## Example Usages
Preprocess the data
```python
# Instantiate Preprocessing and prepare data
preprocess = Preprocessing('game.pgn')
board_states = preprocess.preprocess_data()
```
Then set the Hyperparameters for later training
```python
# Instantiate Hyperparameters
hyperparameters = Hyperparameters()
```
Then make the dataset for training later
```python
# Instantiate makeDataset and prepare dataset
training_data = makeDataset(board_states)
dataset = training_data.prepare_dataset()
```
Create an empty model
```python
# Create a model using the instainited hyperparamters
make_model = Model(hyperparameters,gpu=True)
model = make_model.makeModel()
```
Train the model
```python
# Starting Training
train = Train(dataset, model, hyperparameters, gpu=True)
train.train()
```
To save the model
```python
#Save the model
model_path = 'chess_net.pth'
torch.save(model.state_dict(), model_path)
```
To load the model in another file
```python
from chessNet import ChessNet,Hyperparameters,Model,Postprocessing

hyperparameters = Hyperparameters() #Make the hyperparameters
make_model = Model(hyperparameters,gpu=True)
model = make_model.makeModel() # Make an untrained model
model_path = 'chess_net.pth' # Model saved path
input_state = np.array([-4., -2., -3., -5., -6., -3., -2., -4.,
                        -1., -1., -1., -1., -1., -1., -1., -1.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                         1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                         4.,  2.,  3.,  5.,  6.,  3.,  2.,  4.])
# This is what a starting chess game looks like

post_train = Postprocessing(model,True)
model = post_train.load_model_data(model,model_path) # load the pretrained model into the empty model
print(post_train.predict_next_state(input_state))
# Output the next state
