# Batch Gradient Descent
## Definition

It is a optimization algorithm used for training Machine Learning models. In this varient of gradient descent <b>entire dataset</b> is used to compute gradient of the cost function at each step of the iteration. This means that every update to the model's parameters is based on the gradient calculation fro <b>the full dataset</b>.

## Mathematical Foundation

<b>Step 1 - Initialize Parameters Θ</b> 
<br>
often initialize to small random values or zero.

<b>Step 2 - Compute the Cost Function \J(Θ) :</b>
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} ({Θ}^T x^{(i)} - y^{(i)})^2
$$

<b>Step 3 - Compute the Gradient ∇J(Θ) :
</b>
$$ ∇ J(\theta) = \frac{1}{m} \sum_{i=1}^{m} ({Θ}^T x^{(i)} - y^{(i)})x^{(i)}
$$

<b>Step 4 - Update Parameters : </b>
$$Θ = Θ-α  ∇J(Θ)$$

<b>Step 5 - Repeat</b>

## Pros and Cons
<b>Advantage - </b>
- Convergence Stability
- Efiiciency with Small Datasets

<b>Disadvantages - </b>
- Computationally Expensive
- Memory Usage

# Stochestic Gradient Descent
## Definition 
It is a optimization algorithm used for training Machine Learning models. It updates the parameters using <b>one randomly selected training example </b> at each iteration.

## Advantages

- Faster Convergence
- Less Memory Usage
- Noisy Updates - The updates in SGD are noisy due to the random selection of training examples, which can help in escaping local minima and finding a better global minimum.











# Mini-batch Gradient Descent



## Definition 
It is a ahybrid approach that combines aspects of both batch and stochastic gradient descent. It computes the gradients using small random subsets of the dataseet called <b>mini-batches</b>.

## Advantages
- Efficient Convergence
- Less Noisy Updates
- Parallelism: Allows for efficient parallel processing, as the gradients can be computed simultaneously for different mini-batches.
- scalability: Well-suited for large datasets and can leverage parallel processing for efficient computation. 

## Disadvantages
- Parameter Tuning (mini-batch size)
- Memory Consumption
