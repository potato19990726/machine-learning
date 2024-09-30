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
lr = 0.03
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
gamma = 0.7

# Gradient descent algorithm
def gradient_descent(last_coeffs):
    this_coeffs = np.empty((2,))
    this_coeffs[0] = last_coeffs[0] - lr * np.mean(obj_fun(*last_coeffs) - y)
    this_coeffs[1] = last_coeffs[1] - lr * np.mean((obj_fun(*last_coeffs) - y) * x)
    return this_coeffs

# Momentum gradient descent algorithm
def momentum_gradient_descent(last_coeffs, last_update):
    this_update = np.empty((2,))
    error = obj_fun(*last_coeffs) - y
    this_update[0] = gamma * last_update[0] + lr * np.mean(error)
    this_update[1] = gamma * last_update[1] + lr * np.mean(error * x)
    this_coeffs = last_coeffs - this_update
    return this_coeffs, this_update

def Adam_gradient_descent(last_coeffs, last_m, last_v, t):
    error = obj_fun(*last_coeffs) - y
    g = np.empty((2,))
    g[0] = np.mean(error)
    g[1] = np.mean(error * x)
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
    this_coeffs = gradient_descent(last_coeffs)
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
    this_coeffs, last_update = momentum_gradient_descent(last_coeffs, last_update)
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
    this_coeffs, last_m, last_v = Adam_gradient_descent(last_coeffs, last_m, last_v, j)
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