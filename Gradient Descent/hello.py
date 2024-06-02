import numpy as np
import math
def Batch_Gradient_Descent(
        X, 
        Y, 
        theta,
        centered = True,
        method = "vanilla",
        momentum_coff_1 = 0.9,
        momentum_coff_2 = 0.9999,
        learning_rate = 0.01, 
        max_iter = 1000,
        tol = 1e-6,
        penalty = 'l2',
        alpha = 0.0001,
        l1_ratio = 0.15,
        verbose = 0,
        epsilon = 0,
    ):
    
    # if data is not centered
    if centered == False:
        ones_column = np.ones((100, 1))
        X = np.append(X, ones_column, axis = 1)

    m = len(Y)
    cost_history = []

    if method == "momentum" or "Nesterov_Accelerated_Gradient":
        velocity = np.zeros(np.shape(theta))

    elif method == "AdaGrad" or "RMSprop":
        velocity = np.zeros(np.shape(theta))
        G = 0

    elif method == "Adam":
        velocity = np.zeros(np.shape(theta))    
        momentum = np.zeros(np.shape(theta))
        m_hat = np.zeros_like(momentum)
        v_hat = np.zeros_like(velocity)
        G = 0

    for iteration in range(max_iter):
        
        # compute gradient
        predictions = np.dot(X, theta)
        error = predictions - Y
        gradient = (1/m) * X.T.dot(error)
        
        # method
        if method == "Vanilla":
            error = np.dot(X, theta) - Y
            gradient = (1/m) * np.dot(np._T(X), (error))
            
            # Regularization
            if penalty == 'l2':
                gradient += (alpha / m) * theta
            elif penalty == 'l1':
                gradient += (alpha / m) * np.sign(theta)
            else:
                pass
            
            theta -= learning_rate * gradient

        elif method == "momentum":
            error = np.dot(X, theta)
            gradient = (1/m) * np.dot(np._T(X), (error))
            
            # Regularization
            if penalty == 'l2':
                gradient += (alpha / m) * theta
            elif penalty == 'l1':
                gradient += (alpha / m) * np.sign(theta)
            else:
                pass
            
            velocity = momentum_coff_1 * velocity + (1 - momentum_coff_1) * gradient
            theta -= learning_rate * velocity

        elif method == "Nesterov_Accelerated_Gradient":
            lookahead_theta = theta - learning_rate * momentum_coff_1 * velocity
            error = np.dot(X, lookahead_theta)
            gradient = (1/m) * np.dot(np._T(X), (error))
            
            # Regularization
            if penalty == 'l2':
                gradient += (alpha / m) * theta
            elif penalty == 'l1':
                gradient += (alpha / m) * np.sign(theta)
            else:
                pass
            
            velocity = momentum_coff_1 * velocity + (1 - momentum_coff_1) * gradient
            theta -= learning_rate * velocity

        elif method == "AdaGrad":
            error = np.dot(X, theta)
            gradient = (1/m) * np.dot(np._T(X), (error))
            
            # Regularization
            if penalty == 'l2':
                gradient += (alpha / m) * theta
            elif penalty == 'l1':
                gradient += (alpha / m) * np.sign(theta)
            else:
                pass
            
            G = G + gradient ** 2
            velocity = momentum_coff_1 * velocity + (1 - momentum_coff_1) * (gradient / (math.sqrt(G) + epsilon))
            theta -= learning_rate * velocity

        elif method == "RMSprop":
            error = np.dot(X, theta)
            gradient = (1/m) * np.dot(np._t(X), (error))
            
            # Regularization
            if penalty == 'l2':
                gradient += (alpha / m) * theta
            elif penalty == 'l1':
                gradient += (alpha / m) * np.sign(theta)
            else:
                pass
            
            G = momentum_coff_1 * G + (1 - momentum_coff_1) * (gradient ** 2)
            velocity = momentum_coff_1 * velocity + (learning_rate * gradient) / (math.sqrt(G) + epsilon)
            theta -= learning_rate * velocity

        elif method == "Adam":
            error = np.dot(X, theta)
            gradient = (1/m) * np.dot(np._T(X), (error))
            
            # Regularization
            if penalty == 'l2':
                gradient += (alpha / m) * theta
            elif penalty == 'l1':
                gradient += (alpha / m) * np.sign(theta)
            else:
                pass
            
            momentum = momentum_coff_1 * momentum + (1 - momentum_coff_1) * gradient
            velocity = momentum_coff_2 * velocity + (1 - momentum_coff_2) * (gradient ** 2)
            m_hat = momentum / (1 - momentum_coff_1 ** (iteration + 1))
            v_hat = velocity / (1 - momentum_coff_2 ** (iteration + 1))
            theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # compute cost for the sampled data point
        cost = np.mean(np.square(np.dot(X, theta) - Y))
        cost_history.append(cost)
        
        # print progress
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration} : Cost {cost}")
        
        # check Convergence 
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    return theta, cost_history
