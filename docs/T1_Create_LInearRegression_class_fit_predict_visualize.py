## Create LinearRegression class with fit and predict methods. Visualize train process.

- Create a `LinearRegression` class with a `fit` method to implement the pseudo-code above. Add to your class a `predict` method to make new predictions using the trained model. Test your class with the following example.
    
  ```Python
  # 1. Create synthetic data with noise
  np.random.seed(0)
  X = np.random.rand(100, 1) # array with 100 rows and 1 column (1 feature)
  y = 2 + 3 * X + np.random.randn(100, 1) * 0.1
  # 2. Create and train the model
  model = LinearRegression(learning_rate=0.1, max_iter=1000)
  model.fit(X, y)
  # 3. Make predictions
  X_test = np.array([[0.5]])
  y_pred = model.predict(X_test)
  print(f"Prediction for X=0.5: {y_pred[0]}")
  ```
- Create an animation that shows the position of the fitted line for successive epochs for the example above, similarly to the animation in this [video](https://youtu.be/QoK1nNAURw4).
- How can you adapt the code to address a classification problem where the response $y$ can only be 0 or 1?

