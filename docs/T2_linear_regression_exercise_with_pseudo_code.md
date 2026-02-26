<details>
  <summary>Script: gradient descent with PyTorch, train only, stochastic gradient descent</summary>

```python
# This example illustrates: gradient descent with PyTorch, train only, stochastic gradient descent (SGD)
import matplotlib.pyplot as plt
import torch
import numpy as np
torch.manual_seed(42)

step_size = 0.001  # learning rate
iter = 20  # number epochs

############################################ Creating synthetic data
# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)  # view converts to rank-2 tensor with one column
func = -5 * X + 2
# Adding Gaussian noise to the function f(X) and saving it in Y
y = func + 0.4 * torch.randn(X.size())

########################################## Baseline: Linear regression LS solution
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
print('Least square LR coefficients:',reg.intercept_,reg.coef_)

####################################################### Gradient Descent
# initial weights
coeffs = torch.tensor([-20., -10.], requires_grad=True)

# defining the function for prediction (linear regression)
def calc_preds(x):
    return coeffs[0] + coeffs[1] * x

# Computing MSE loss for one example
def calc_loss_from_labels(y_pred, y):
    return torch.mean((y_pred - y) ** 2) # MSE

# lists to store losses for each epoch
training_losses = []

# epochs
for i in range(iter):
    # calculating loss as in the beginning of an epoch and storing it
    y_pred = calc_preds(X)
    loss = calc_loss_from_labels(y_pred, y)
    training_losses.append(loss.item())

    # Stochastic Gradient Descent (SGD): update weights 
    for j in range(X.shape[0]):
        # randomly select a data point
        idx = np.random.randint(X.shape[0])
        x_point = X[idx]
        y_point = y[idx]

        # making a prediction in forward pass
        y_pred = calc_preds(x_point)

        # calculating the loss between predicted and actual values
        loss = calc_loss_from_labels(y_pred, y_point)

        # compute gradient
        loss.backward()

        # update coeffs
        with torch.no_grad():
            coeffs.sub_(coeffs.grad * step_size)
            # zero gradients
            coeffs.grad.zero_() # prevents from accumulating

print('coeffs found by stochastic gradient descent:', coeffs.detach().numpy())

# plot training loss along epochs
plt.plot(training_losses, '-g')
plt.xlabel('epoch')
plt.ylabel('loss (MSE)')
plt.show()
```
</details>

--->
