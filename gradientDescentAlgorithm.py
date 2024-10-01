import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# The data to fit
m = 100
bias_true = 2
weight_true = 0.5
x = np.linspace(-3, 3, m)
y = bias_true + weight_true * x + np.random.randn(m) * 0.5

def obj_fun(bias, weight):
    return weight * x + bias

def mse(bias, weight):
    if np.isscalar(weight):
        return np.mean((y - obj_fun(bias, weight))**2)
    else:
        # temp_w = weight[:, np.newaxis] create a 50x1 matrix (newaxis is amazing)
        return np.average((y - obj_fun(bias[np.newaxis,:,np.newaxis], weight[:,np.newaxis,np.newaxis]))**2, axis=2)
    
# Plot of objective function
fig = plt.figure(figsize=(10, 6.15))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')
ax1.scatter(x, y, marker='x', s=10, color='k')
# Plot of objective function

# Prepare the background for gradient descent
bias_grid = np.linspace(-0.3,3.5,101)
weight_grid = np.linspace(0,1,101) # Note that there are 101 points here
possible_error = mse(bias=bias_grid, weight=weight_grid)
X, Y = np.meshgrid(bias_grid, weight_grid)
contours = ax2.contour3D(X, Y, possible_error, 50)
# ax2.clabel(contours)
# The target parameter values indicated on the cost function contour plot
# ax2.scatter([bias_true]*2,[weight_true]*2,s=[50,10], color=['k','w'])
ax2.set_ylabel(r'$\omega$')
ax2.set_xlabel(r'$b$')
ax2.set_zlabel('Cost')
ax2.set_title('Cost function')
# Prepare the background for gradient descent

# Variables for gradient descent
N = 50
# lr = 0.03
lr_gd = 0.05
lr_momentum = 0.03
lr_adam = 0.05
beta1_adam = 0.95
beta2_adam = 0.99
epsilon = 1e-8
gamma_momentum = 0.7

# Gradient descent algorithm
def gradient_descent(last_coeffs, lr):
    this_coeffs = np.empty((2,))
    this_coeffs[0] = last_coeffs[0] - lr * np.mean(obj_fun(*last_coeffs) - y)
    this_coeffs[1] = last_coeffs[1] - lr * np.mean((obj_fun(*last_coeffs) - y) * x)
    return this_coeffs

# Momentum gradient descent algorithm
def momentum_gradient_descent(last_coeffs, last_update, lr, gamma):
    this_update = np.empty((2,))
    error = obj_fun(*last_coeffs) - y
    this_update[0] = gamma * last_update[0] + lr * np.mean(error)
    this_update[1] = gamma * last_update[1] + lr * np.mean(error * x)
    this_coeffs = last_coeffs - this_update
    return this_coeffs, this_update

def Adam_gradient_descent(last_coeffs, last_m, last_v, t, lr, beta1, beta2):
    error = obj_fun(*last_coeffs) - y
    g = np.empty((2,))
    g[0] = lr * np.mean(error)
    g[1] = lr * np.mean(error * x)
    this_m = beta1 * last_m + (1 - beta1) * g
    this_v = beta2 * last_v + (1 - beta2) * (g ** 2)
    m_hat = this_m / (1 - beta1 ** t)
    v_hat = this_v / (1 - beta2 ** t)
    this_coeffs = last_coeffs - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return this_coeffs, this_m, this_v


# Run gradient descent algorithm
coeffs =[np.array((0, 0))] # In line with mse(bias, weight), bias is placed in front [0], weight is placed behind [1]
all_mse = [mse(*coeffs[0])] # Using * is very convenient
for j in range(1, N):
    last_coeffs = coeffs[-1]
    this_coeffs = gradient_descent(last_coeffs, lr_gd)
    coeffs.append(this_coeffs)
    all_mse.append(mse(*this_coeffs))
final_bias, final_omega = coeffs[-1]
print('Gradient Descent:')
print(f"Final bias (b): {final_bias:.3f}")
print(f"Final omega (ω): {final_omega:.3f}")

# Run momentum gradient descent algorithm
momentum_coeffs =[np.array((0, 0))] # In line with mse(bias, weight), bias is placed in front [0], weight is placed behind [1]
momentum_all_mse = [mse(*momentum_coeffs[0])] # Using * is very convenient
last_update = np.array((0, 0))
for j in range(1, N):
    last_coeffs = momentum_coeffs[-1]
    this_coeffs, last_update = momentum_gradient_descent(last_coeffs, last_update, lr_momentum, gamma_momentum)
    momentum_coeffs.append(this_coeffs)
    momentum_all_mse.append(mse(*this_coeffs))
momentum_final_bias, momentum_final_omega = momentum_coeffs[-1]
print('Momentum Gradient Descent:')
print(f"Final bias (b): {momentum_final_bias:.3f}")
print(f"Final omega (ω): {momentum_final_omega:.3f}")

# Run Adam gradient descent algorithm
adam_coeffs =[np.array((0, 0))] # In line with mse(bias, weight), bias is placed in front [0], weight is placed behind [1]
adam_all_mse = [mse(*adam_coeffs[0])] # Using * is very convenient
last_m = np.array((0, 0))
last_v = np.array((0, 0))
for j in range(1, N):
    last_coeffs = adam_coeffs[-1]
    this_coeffs, last_m, last_v = Adam_gradient_descent(last_coeffs, last_m, last_v, j, lr_adam, beta1_adam, beta2_adam)
    adam_coeffs.append(this_coeffs)
    adam_all_mse.append(mse(*this_coeffs))
adam_final_bias, adam_final_omega = adam_coeffs[-1]
print('Adam Gradient Descent:')
print(f"Final bias (b): {adam_final_bias:.3f}")
print(f"Final omega (ω): {adam_final_omega:.3f}")

# Animation function
def update(frame):
    ax2.clear()
    ax2.contour3D(X, Y, possible_error, 100)
    ax2.set_ylabel(r'$\omega$')
    ax2.set_xlabel(r'$b$')
    ax2.set_zlabel('Cost')
    ax2.set_title('Cost function')
    if frame < N:
        ax2.scatter(*zip(*coeffs[:frame+1]), c='blue', s=30, lw=0, zs=all_mse[:frame+1])
        ax2.scatter(*zip(*momentum_coeffs[:frame+1]), c='red', s=30, lw=0, zs=momentum_all_mse[:frame+1])
        ax2.scatter(*zip(*adam_coeffs[:frame+1]), c='green', s=30, lw=0, zs=adam_all_mse[:frame+1])
    else:
        ax2.scatter(*zip(*coeffs), c='blue', s=30, lw=0, zs=all_mse)
        ax2.scatter(*zip(*momentum_coeffs), c='red', s=30, lw=0, zs=momentum_all_mse)
        ax2.scatter(*zip(*adam_coeffs), c='green', s=30, lw=0, zs=adam_all_mse)

ani = FuncAnimation(fig, update, frames=range(N+1), repeat=False)
ani.save('gradient_descent_animation.gif', writer=PillowWriter(fps=10))
plt.show()

def run_optimization(optimizer, lr, gamma=None, beta1=None, beta2=None, n_iterations=50):
    coeffs = [np.array((0, 0))]
    all_mse = [mse(*coeffs[0])]
    
    if optimizer == 'gd':
        for _ in range(1, n_iterations):
            coeffs.append(gradient_descent(coeffs[-1], lr))
            all_mse.append(mse(*coeffs[-1]))
    elif optimizer == 'momentum':
        last_update = np.array((0, 0))
        for _ in range(1, n_iterations):
            this_coeffs, last_update = momentum_gradient_descent(coeffs[-1], last_update, lr, gamma)
            coeffs.append(this_coeffs)
            all_mse.append(mse(*this_coeffs))
    elif optimizer == 'adam':
        last_m = np.array((0, 0))
        last_v = np.array((0, 0))
        for j in range(1, n_iterations):
            this_coeffs, last_m, last_v = Adam_gradient_descent(coeffs[-1], last_m, last_v, j, lr, beta1, beta2)
            coeffs.append(this_coeffs)
            all_mse.append(mse(*this_coeffs))
    
    return coeffs[-1], all_mse[-1]

def grid_search():
    learning_rates = np.linspace(0.01, 0.3, 50)
    gammas = np.linspace(0.5, 0.95, 50)
    beta1s = np.linspace(0.8, 0.99, 20)
    beta2s = np.linspace(0.8, 0.999, 20)
    
    best_params = {}
    
    # Gradient Descent
    best_mse_gd = float('inf')
    for lr in learning_rates:
        final_coeffs, final_mse = run_optimization('gd', lr)
        if final_mse < best_mse_gd:
            best_mse_gd = final_mse
            best_params['gd'] = {'lr': lr}
    
    # Momentum
    best_mse_momentum = float('inf')
    for lr in learning_rates:
        for gamma in gammas:
            final_coeffs, final_mse = run_optimization('momentum', lr, gamma=gamma)
            if final_mse < best_mse_momentum:
                best_mse_momentum = final_mse
                best_params['momentum'] = {'lr': lr, 'gamma': gamma}
    
    # Adam
    best_mse_adam = float('inf')
    for lr in learning_rates:
        for beta1 in beta1s:
            for beta2 in beta2s:
                final_coeffs, final_mse = run_optimization('adam', lr, beta1=beta1, beta2=beta2)
                if final_mse < best_mse_adam:
                    best_mse_adam = final_mse
                    best_params['adam'] = {'lr': lr, 'beta1': beta1, 'beta2': beta2}
    
    print("Best parameters for Gradient Descent:", best_params['gd'])
    print("Best MSE for Gradient Descent:", best_mse_gd)
    print("Best parameters for Momentum:", best_params['momentum'])
    print("Best MSE for Momentum:", best_mse_momentum)
    print("Best parameters for Adam:", best_params['adam'])
    print("Best MSE for Adam:", best_mse_adam)
    return best_params

# Run grid search
best_params = grid_search()

# Visualize the results for each algorithm
plt.figure(figsize=(18, 6))

for i, (optimizer, params) in enumerate(best_params.items()):
    plt.subplot(1, 3, i+1)
    
    if optimizer == 'gd':
        final_coeffs, _ = run_optimization(optimizer, params['lr'])
    elif optimizer == 'momentum':
        final_coeffs, _ = run_optimization(optimizer, params['lr'], gamma=params['gamma'])
    else:  # 'adam'
        final_coeffs, _ = run_optimization(optimizer, params['lr'], beta1=params['beta1'], beta2=params['beta2'])

    plt.scatter(x, y, color='blue', label='Data points', alpha=0.5)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = final_coeffs[1] * x_line + final_coeffs[0]
    plt.plot(x_line, y_line, color='red', label='Final fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Final fit using {optimizer.upper()} optimizer')
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('bestAlgorithmResults.png')
plt.close()