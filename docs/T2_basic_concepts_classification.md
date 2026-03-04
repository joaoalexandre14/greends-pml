This notebook includes the following topics:
- Classification and logistic regression
- Entropy and cross-entropy
- Regularization
- Batch size 

<details markdown="block">
<summary> Binary classification and the logistic function </summary>

Since linear regression predicts a continuous value that is unbounded, it is not adequate for *classification* problems. Let's consider the simple case of a *binary classification problem*, e.g. predict the sex from other variables for the Penguin data set. 

In that case, the are just two classes *female* and *male*, and we only need to consider one of them in the model. What we intend to predict is the **probability** that a new example is of class *female* (if it is, say, 78% then the probability of being of class *male* will just be 22%). 

We need a function that converts the output of the linear regression $y$ into a probability. This is done by the logistic function:

$$\sigma(y)=\frac{1}{1+e^{-y}}.$$

For instance, for 3 predictor variables, the full  *logistic regression* prediction model for binary classification is the following:

$$\hat{p}= \frac{1}{1+e^{-(w_0 + w_1 \\, x_1 +  w_2 \\, x_2 + w_3 \\, x_3)}}.$$

where $\hat{p}$ is the estimated probability that the example belongs to the class. Let's consider the example above where the class is *female*. If $\hat{p} \ge 0.5$ then, the predicted class $\hat{y}$ is *female*. Otherwise, the predicted class $\hat{y}$ is *male*. To fit the model, one also solves the ML minimization problem where the loss function depends for each example on the true label $y \in \{0,1\}$ and the predicted probability $\hat{p} \in [0,1]$. 

The loss function should be low if $\hat{p}$ is close to 1 and the true class is *female* or if $\hat{p}$ is close to 0 and the true class is *male*. This typical loss function for this kind of problem is known as *log loss*,  *binary cross-entropy* or *logistic loss* and it is defined by $L=-[y {\rm log}(\hat{p}) + (1-y) {\rm log}(1-\hat{p})]$ where $y$ is the true label.

</details>

<details markdown="block">
<summary> Entropy and cross-entropy </summary>

</details>


<details markdown="block">
<summary> Regularization to avoid overfitting</summary>

</details>

