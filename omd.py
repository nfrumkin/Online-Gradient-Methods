import numpy as np
import matplotlib.pyplot as plt
    
class omd:
    def __init__(self, dim, eta):
        self.dim = dim
        self.x_t = (1.0/dim)*np.ones([dim])
        self.t = 0
        self.eta = eta
    
    def step(self, loss_grad):

        self.t = self.t + 1

        # compute numerator
        y_t = self.x_t*np.exp(-1*self.eta*(loss_grad))

        # compute denominator
        normalizer = np.sum(np.abs(y_t))
        self.x_t = y_t/normalizer

        return self.x_t

def generate_simplex(dim):
    p = np.random.uniform(0,100, size=[dim])
    return p/np.sum(p)

def loss(x,y, l="l2"):
    if l == "l2":
        return 0.5*np.linalg.norm(x-y)
    elif l == "l1":
        return np.sum(np.abs(x-y))
    elif l == "kl":
        # sum over the support of y
        return np.sum(np.where(x != 0,(x-y) * np.log(x / y), 0))
    elif l == "linear":
        return np.sum(x-y)

def grad_loss(x,y, l="l2"):
    if l == "l2":
        return x - y
    elif l == "l1":
        return np.sum(np.sign(x-y))
    elif l == "kl":
        return np.ones(x.shape) - np.divide(y,x)#np.divide(x,y)+np.log(np.divide(x,y))
    elif l == "linear":
        return -1*x.shape[0]

if __name__ == "__main__":
    dim = 5          # dimension
    eta = 0.1                   # stepsize
    T = 700                     # number of steps
    losses = np.zeros([T,1])    # loss values placeholder
    threshold = 0.0001          # convergence threshold
    loss_func = "linear"                    # choose loss function

    p = generate_simplex(dim)
    # p = np.array([0.19775466, 0.16387309, 0.22701363, 0.10678895, 0.30456967])
    # p = p.T

    online_md = omd(dim, eta)
    # determine initial value
    x_init = online_md.x_t
    x_t = x_init

    for t in range(0,T):
        #print(x_t)
        x_t = online_md.step(grad_loss(x_t, p, loss_func))
        loss_t = loss(x_t, p, loss_func)
        losses[t] = loss_t
        
        # check for convergence
        norm_dist = np.linalg.norm(p - x_t)
        if norm_dist < threshold:
            print("solution converged at iter ", t)
            break
    
    plt.plot(losses)
    plt.ylabel("loss")
    plt.xlabel("iter")
    plt.savefig("plots/losses.png")

    print("initialization:\t\t", x_init)
    print("objective simplex:\t", p)
    print("learned simplex:\t", x_t)
    print("norm distance: ", norm_dist)

