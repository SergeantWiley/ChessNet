
## Break down of the lessson

- Architecture and Hyperparamaters Recap
- Multidimensional Mathematics
- Crazy Math!

## Architecture and Hyperparamaters Recap

As a recap, our python code for the machine learning itself is below. 
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

# Define the hyperparameters
input_size = 64  # Size of the input tensor
hidden_size = 128  # Size of the hidden layer
output_size = 64  # Size of the output tensor
learning_rate = 0.001
batch_size = 8
num_epochs = 50
```
## Multidimensional Mathematics

Its recommended to have a understanding of Multidimensional Mathematics before procceeding. Specifically vectors and matrices

Vecotrs signify direction and magnitude and in our case, a vector is a chess peice. The matrix itself is the chess board. While a chess board seems like a 2D enviroment, for a machine learning algorithm to understand it has to be in higher dimensions
```bash
-4 -2 -3 -5 -6 -3 -2 -4
-1 -1 -1 -1 -1 -1 -1 -1
0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0
1  1  1  1  1  1  1  1
4  2  3  5  6  3  2  4
```
This looks 2D for anyone unfamiliar to matrices but this is in fact in 8 dimensions. Each value is a chess piece to us but its a vector far as machine learning is concerned and this is critical to determine the two transformation matrices. With this in mind, its important to know that each formula below interact with one position (nueron) within our matrix.

With that in mind, we will begin our first linear transformation matrix function.

## Transformations

Our input_tensor (matrix) is $`x`$ which is transformed in to a 1x64 vector. During the training, $`W_1`$ and $`b_1`$ are adjusted with our Adam optimizer. The Adam Optimizer will be described later in the meantime, our first layer (`fc1`) looks like this

$`
z^{(1)} = x \cdot W_1 + b_1
`$

To visualize the matrices, we can look below

$`x = \begin{bmatrix}x_{1,1} & x_{1,2} & \dots & x_{1,64} \\\end{bmatrix}`$

$`W_1 = \begin{bmatrix}
W_{1,1,1} & W_{1,1,2} & \dots & W_{1,1,128} \\
W_{1,2,1} & W_{1,2,2} & \dots & W_{1,2,128} \\
\vdots & \vdots & \ddots & \vdots \\
W_{1,64,1} & W_{1,64,2} & \dots & W_{1,64,128} \\
\end{bmatrix}`$

$`b_1 = \begin{bmatrix}
b_{1,1} & b_{1,2} & \dots & b_{1,128} \\
\end{bmatrix}`$

$`z_1 = \begin{bmatrix}x_{1,1} & x_{1,2} & \dots & x_{1,64} \\\end{bmatrix} \cdot \begin{bmatrix}
W_{1,1,1} & W_{1,1,2} & \dots & W_{1,1,128} \\
W_{1,2,1} & W_{1,2,2} & \dots & W_{1,2,128} \\
\vdots & \vdots & \ddots & \vdots \\
W_{1,64,1} & W_{1,64,2} & \dots & W_{1,64,128} \\
\end{bmatrix} + \begin{bmatrix}
b_{1,1} & b_{1,2} & \dots & b_{1,128} \\
\end{bmatrix}`$

Quite a lot. Thats not even the second layer. The second layer has to get pass into a Relu (Rectified Linear Unit) Function but mathematically it looks way simpler. To descibe the ReUL function lightly, it converts each value into a value ranging from 0 to 1

$`
ReLU(z_1) = \begin{cases}
\text{if } z \geq 0 \\
0 \text{if } z < 0
\end{cases}
`$

$`
ReLU(z-1) = a_1
`$

Now its passing it to the next layer (`fc1`). This new matrix ($`a^{(1)}`$) is sent to 

$`
z^{(2)} = a_1 \cdot W_1 + b_1
`$

Since our tranformations end after we get $`z_2`$ however since there are no more layers, the next attention ($`a_2`$) is set to be equal $`z_2`$ and this is our output matrix. We then use the code below to output
```python
def decode_output(output):
    return np.round(output.view(8, 8).detach().cpu().numpy()).astype(int)
```
However, this doesnt show the math behind the training as this assumes we already know the weights. The next part is quite scary but will be dumbed down to the best of our ability.

To train our model, we use the Adam optimizer which is the most common. This optimizer works in super high dimensions. First our vectors are set to 0. Our vectors are $`m`$ and $`v_0`$

$`m = 0`$

$`v_1 = 0`$

Our Gradaints look like this

$`\nabla_0J(\Theta)`$

While it looks terrifying, this is basically just the weights ($`W`$) and bias ($`b`$) from before. They get passed into our Moment Estimates

$`m_t = \beta_1\cdot m_{t-1}+(1+\beta_1)\cdot \nabla_0J(\Theta_t)`$

$`v_t = \beta_1\cdot v_{t-1}+(1+\beta_1)\cdot (\nabla_0J(\Theta_t))^2`$


To simplfy this, we take an exponetial decay which is usally 0.9 to 0.999 but may very depending on the percision we need and reference it to our past vectors as well as parameters. These values will then be passed to a bias correction whucg is a standard procedure used in stastistics

$`m_t = \frac{{m_t}}{{1-\mathrm{\beta}_{t}^{1}}}`$

$`v_t = \frac{{v_t}}{{1-\mathrm{\beta}_{t}^{1}}}`$

The part above is just an adjustment to add additional integrity so its not required to know but its one of those things that are nice to have. Once we have done that, we update our parameters to get a closer. The next part is what makes the learning in machine learning

$`θ_{t+1}=θ_t​− \frac{η}{v_t+ϵ}⋅m_t`$

While this formula may also look scary due to some new characters rather its realitively simple. Heres a simplifed break down

Our new parameter we will pass back up to our $`\nabla_0J(\Theta)`$  to repeat the proccess cycle through a new iteration. Our current paramters are subtractracted by our learning_rate ($`{η}`$) hyperparameter is divided by our next $`v_t`$ vector. The wierd looking symbol $`ϵ~`$ is a very small value (around $`10^{-8}`$) prevent divide by 0 errors. Then our current $`m_t`$ vector is multipled to get a new $`\Theta`$. 

The final part. Epochs. They are the only thing we havent mentioned. A side note about our wierd $`{η}`$ character. This is the learning_rate parameter we declared. Within one epoch we have many iterations ($`{t}`$). To find the amount of iterations, its pretty simple. 

```python
t = dataset_size/batch_size
``` 

This can be valueable since a smaller batch_size means more time ChessNet can learn but one epoch never actually resets or passes any values to the next epoch rather its a unit of time we limit the machine learning algorithm to have. Its the ML equlivent to a deadline. 


The goal to is not to understand any of this let alone all of it but get a concept of the complexities that happen behind the scene so it gives us an appreciation of the libraries we have to simplfy our work. 

