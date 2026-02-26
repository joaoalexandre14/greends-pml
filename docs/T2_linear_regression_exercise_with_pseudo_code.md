## Exercise with pseudo-code for SGD

Consider the following pseudo-code to train a simple Linear Regression model. What is the *loss* function that we aim at minimizing? What is the strategy to reduce the *loss* in each iteration? Is there a risk of *over-fitting*?
  
  ---
  Pseudo code for SGD (stochastic gradient descent) to fit a linear regression:
  
  - Dataset:  $D = {(x_1^{(i)}, ..., x_n^{(i)}, y^{(i)})}\_{i=1}\^N$  `N observations, n features`
  - Learning rate:  $\eta$ `Small positive value`
  - Max iterations: max_iter `Number of epochs`
  - Initial weights $w$ := $(w_0, w_1, ..., w_n)$ `Typically, all zero`
  - For iter := 1 to max_iter 
    - For each  $(x_1, ..., x_n, y) \in D$  `Update weights after each example`
      - $\hat{y}$ := $w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$ `Predict response with current weights`
      - error := $y-\hat{y}$
      - $w_0$ := $w_0 + \eta \cdot$ error # `Update weight (bias)`
      - For $j$ := 1 to $n$
        - $w_j$ := $w_j + \eta \cdot$ error $\cdot x_j$ # `Update weight (for each feature)`
          
  ---

- Create a `LinearRegression` class with a `fit` method to implement the pseudo code above. Add to your class a `predict` method to make new predictions using the fitted model. Test your class with the following example.
    
  ```Python
  # Create synthetic data
  np.random.seed(0)
  X = np.random.rand(100, 1) # array with 100 rows and 1 column (1 feature)
  y = 2 + 3 * X + np.random.randn(100, 1) * 0.1
  # Create and train the model
  model = LinearRegression(learning_rate=0.1, max_iter=1000)
  model.fit(X, y)
  # Make predictions
  X_test = np.array([[0.5]])
  y_pred = model.predict(X_test)
  print(f"Prediction for X=0.5: {y_pred[0]}")
  ```
- Create an animation that shows the position of the fitted line for successive epochs for the example above.
- How can you adapt the code to address a classification problem where the response $y$ can only be 0 or 1?

</details>
