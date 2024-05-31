import numpy as np

"""
    Funtionalities:-     

    Batch Gradient Descent (BGD)
        the entire dataset is used to compute the gradient of the cost function at each step of the iteration.

    Stochestic Gradient Descent (SGD)
        SGD makes updates to the model parameters using only a single training example at a time.

    Mini-batch Gradient Descent (MBGD)
        It computes the gradients using small random subsets of the dataseet called mini-batches.
"""
"""
    Parameters:- 


    X : Features array
    
    Y : Target Variable
    
    theta : parameters to be learned
    
    learning_rate : steps taken every iteration in opposite direction of gradient
        - By- Default value = 0.01
    
    max_iter : maximum number of iterations allowed
        - By- Default value = 1000 
    
    tol : after this value "tolerance" errors becoe permissible
        - By- Default value = 1e-6

    penalty : regularization term added to the cost function
        - By- Default value = 'l2'
        - permissible values
            - 'l2'- L2 Regularization 
            - 'l1'- L1 Regularization
            - 'elasticnet' - For Elastic Net Penalty (a combinition of L1 and L2 penalties)
    
    virbose : detail in output messages.
        - By- Default value = 0
        - permissible values
            - 0 - no verbose
            - 1 - less friquent verbose with limited information
            - >1 - more frequent verbose with detailed information

    l1_ratio :  The Elastic Net mixing parameter
        - By default value = 0.15
        - permissible values
            - 0 <= l1_ratio <= 1
            - l1_ratio == 0 : Pure L2 Regularization
            - l1_ratio == 1 : Pure L1 Regularization
"""

def compute_cost(X, Y, theta, penalty = 'l2', alpha = 0.0001):

    m = len(Y)
    prediction = np.dot(X, theta)
    cost = (np.sum(np.square(prediction - Y))) /(2*m)
    if penalty == 'l2':
        cost += (alpha / (2 * m)) * np.sum(np.square(prediction - Y))
    elif penalty == 'li':
        cost += (alpha / (2 * m)) * np.sum(np.abs(theta))
    return cost

def BGD(
        X, 
        Y, 
        theta, 
        learning_rate = 0.01, 
        max_iter = 1000,
        tol = 1e-6,
        penalty = 'l2',
        alpha = 0.0001,
        l1_ratio = 0.15,
        verbose = 0
        ):
    

    m = len(Y)
    cost_history = []
    
    for iteration in range(max_iter):
        
        # compute gradient
        predictions = np.dot(X, theta)
        error = predictions - Y
        gradient = (1/m) * X.T.dot(error)

        # Regularization
        if penalty == 'l2':
            gradient += (alpha / m) * theta
        elif penalty == 'l1':
            gradient += (alpha / m) * np.sign(theta)
        else:
            pass

        # update parameter
        theta -= learning_rate * gradient

        # compute cost for the sampled data point
        cost = compute_cost(X, Y, theta, penalty, alpha)
        cost_history.append(cost)
        
        # print progress
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration} : Cost {cost}")
        
        # check Convergence 
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Converged after {iteration} iterations.")

        return theta, cost_history


def SGD (
        X,
        Y,
        theta,
        learning_rate = 0.01,
        max_iter = 1000,
        tol = 1e-6,
        penalty = 'l2',
        alpha = 0.0001,
        l1_ratio = 0.15,
        verbose = 0
):
    
    m = len(Y)
    cost_history = []

    for iteration in range(max_iter):
        cost = 0

        for i in range(m):

            # sample a random data point
            random_index = np.random.randint(m)
            x_i =X[random_index]
            y_i = Y[random_index]

            # compute gradient for sampled data point
            prediction = np.dot(x_i, theta)
            error = prediction - y_i
            gradient = x_i * error

            # Regularization
            if penalty == 'l2':
                gradient += (alpha / m) * theta
            elif penalty == 'l1':
                gradient += (alpha / m) * np.sign(theta)
            else:
                pass

            # update parameters
            theta -= learning_rate * gradient

            # compute cost for the sampled data point
            cost += 0.5 * (error ** 2)

        # Compute average cost
        cost /= m
        cost_history.append(cost)

        # print progress
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost {cost}")

        # check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Converged aftere {iteration} iterations.")
            break

    return theta, cost_history

def MBGD(
        X,
        Y,
        theta,
        batch_size = 32,
        learning_rate = 0.01,
        max_iter = 1000,
        tol = 1e-6,
        penalty = 'l2',
        alpha = 0.0001,
        verbose = 0
):
    
    m = len(Y)
    cost_history = []

    for iteration in range(max_iter):
        cost = 0

        # shuffle the data indices
        indices = np.random.permutation(m)

        for i in range(0, m, batch_size):

            # select mini-batch
            batch_indices = indices[i : i+batch_indices]   
            x_batch = X[batch_indices]
            y_batch = Y[batch_indices]

            # compute gradient for the mini-batch
            prediction = np.dot(x_batch, theta)
            errors = prediction - y_batch
            gradients = np.dot(x_batch.T, errors) / batch_size

            # Regularization
            if penalty == 'l2':
                gradients += (alpha / m) * theta
            elif penalty == 'l1':
                gradients += (alpha / m) * np.sign(theta)

            # Update parameters
            theta -= learning_rate * gradients
            
            # Compute cost for the mini-batch
            batch_cost = 0.5 * np.mean(errors ** 2)
            cost += batch_cost
        
        # Compute average cost
        cost /= (m // batch_size)
        cost_history.append(cost)
        
        # Print progress
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost {cost}")
        
        # Check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Converged after {iteration} iterations.")
            break
    
    return theta, cost_history 
