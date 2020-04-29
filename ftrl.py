import numpy as np
import matplotlib.pyplot as plt

def ftrl(T, df, eta, dim):
    """
    T  : number of time steps
    df : derivative of the loss function
    eta: positive learning rate
    dim: number of dimentions of x
    The regularizer here is R(x) = 1/2 * ||x||^2
    """

    x = np.zeros(dim)
    for t in range(1, T+1):
        x = x - eta * df(x)

    return x

def ftrl_prob(T, p_optimal, eta, loss='KL'):
    """
    T : number of time steps
    p_optimal : optimal probability distribution (np array)
    eta : postive learning rate
    loss: either KL or l2

    Regularizer is the negative entropy
    loss funtion is KL(p_optimal || p)
    gradient of the loss funtion is (- p_optimal / p)
    """
    
    dim = len(p_optimal)
    p = (1/dim) * np.ones(dim) 
    k = np.zeros(dim)
    one = np.ones(dim)
    losses = []
    if loss == 'KL':
        f = KL
    elif loss == 'l2':
        f = l2
    else:
        raise Exception('Unknown loss')

    for t in range(T):
        if loss == 'KL':
            k = k + eta * (one - (p_optimal / p))
        elif loss == 'l2':
            k = k + eta * (p - p_optimal)
        
        p_ = np.exp(-k) # dummy variable to avoid double computation
        p = p_ / np.sum(p_)
        losses.append(f(p_optimal, p))

    return p, losses

def KL(x, y):
    return np.sum(np.where(x != 0,(x-y) * np.log(x / y), 0))

def l2(p, q):
    return 0.5 * np.linalg.norm(p - q)**2


np.set_printoptions(precision=3, suppress=True)
# p_optimal = np.array([0.15, 0.05, 0.05, 0.5, 0.25])
# Generate a random optimal probability vector
dim = 5
p_optimal_ = np.random.rand(dim)
p_optimal = p_optimal_ / np.sum(p_optimal_)
print(p_optimal)

# Run the FTRL algorithm
loss = 'l2'
p, losses = ftrl_prob(50, p_optimal, 0.1, loss)
print(p)

regret = [sum(losses[:i]) for i in range(len(losses))]
avg_regret = sum(regret)/len(regret)
print('average regret = ', avg_regret)

plt_loss = plt.figure(1)
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('iter')

plt_regret = plt.figure(2)
plt.plot(regret)
plt.ylabel('regret')
plt.xlabel('iter')

plt.show()



