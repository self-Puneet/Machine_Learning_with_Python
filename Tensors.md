<!-- #Tensor Handling with TensorFlow -->
## Introduction to Tensors

Tensors are fundamental data structures in TensorFlow, representing multi-dimensional arrays or matrices. They are the primary building blocks used for storing and manipulating data in TensorFlow computations. Understanding tensors is essential for effectively working with TensorFlow and building machine learning


## Importing Libraries
- tensorflow
- numpy

```python 
import tensorflow as tf
import numpy as np
```

## Creating Tensors

1. Converting **Python list** and **numpy array** into tensor.

```python
python_list = [1, 2, 3, 4, 5]
tensor = tf.constant(python_list)
```

```python
numpy_array = np.array([1, 2, 3, 4, 5])
tensor = tf.constant(numpy_array)
```

2. Creating Tensors with Specific Values. 

  - parameters 
      - shape of tensor
      
```python
tensor_zeros = tf.zeros([3, 3])

tensor_ones = tf.ones([2, 4])
```

3. Creating Tensors with Random Values (Normal Distribution)
  - parameters
    - shape of tensor
    - mean of distribution
    - standard deviation of distribution

```python
tensor_random_normal = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
```

4. Creating Tensors with Random Values (Uniform Distribution)
  - parameters
    - shape of tensor
    - minimum value of distribution
    - maximum value of distribution

```python
tensor_random_uniform = tf.random.uniform([2, 2], minval=0, maxval=10)
```


## Types of Tensors
In TensorFlow, tensors can be categorized based on mutability into two main types:
1. Mutable Tensors - These tensors are mutable, meaning their values can be changed during program execution.

```python
tensor = tf.Variable ([1, 2, 3, 4, 5])
```
2. Immutable Tensors - These tensors are immutable, meaning their values cannot be changed after creation.

```python
tensor = tf.constant ([1, 2, 3, 4, 5])
```


## Opearations on Tensors

1. Accessing Tensor Elements

```python
tensor = tf.constant([[1, 2, 3],[4, 5, 6]])
print(tensor[0,1])
```
2. Reshaping Tesnors

```python
reshaped_tensor = tf.reshape(tensor, [3, 2])
```
3. Concatenating Tensors

```python
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

concatenated_tensor = tf.concat([tensor1, tensor2], axis=0)
```

4. Splitting Tensors - Splitting tensors divides a tensor into multiple smaller tensors along a specified axis.

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

split_tensors = tf.split(tensor, num_or_size_splits=3, axis=1)
```

5. Element-wise Operations - these operations are performed between corresponding elements of 2 tensors.

```python
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

# directly
elementwise_sum = tensor1 + tensor2
elementwise_product = tensor1 * tensor2

# using functions
A = tf.add(tensor1, tensor2)
B = tf.product(tensor1, tensor2)
```
6. Reduction Operations - reduce whole tensor into a number by some operation for eg. adding all elements of tensor, getting maximum element of tensor.

```python
tensor = tf.constant([[1, 2], [3, 4]])

sum_tensor = tf.reduce_sum(tensor)
max_tensor = tf.reduce_max(tensor)
```

7. Mathematical Functions - functions that operate element-wise on tensors.

```python
# Mathematical functions
tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

# Compute the square root of each element
sqrt_tensor = tf.sqrt(tensor)

# Compute the exponential of each element
exp_tensor = tf.exp(tensor)
```

8. Matrix Operations - like matrix multiplication, transposition, etc.

```python
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Matrix multiplication
matrix_product = tf.matmul(matrix, matrix)

# Matrix transposition
transposed_matrix = tf.transpose(matrix)

# Matrix Inversion
inverse_matrix = tf.linalg.inv(square_matrix)

# Matrix Determinant
determinant = tf.linalg.det(square_matrix)

# Matrix Trace
trace = tf.linalg.trace(square_matrix)
```
