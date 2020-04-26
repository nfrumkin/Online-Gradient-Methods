import numpy as np
import matplotlib.pyplot as plt

class omd:
    def __init__(self, dim, eta):
        self.x_t = (1.0/dim)*np.ones([dim])
        self.t = 0
        self.eta = eta
    
    def step(self, loss_grad):

        self.t = self.t + 1

        # compute numerator
        self.x_t = self.x_t*np.exp(-1*self.eta*loss_grad)

        # compute denominator
        normalizer = np.sum(self.x_t)
        self.x_t = self.x_t/normalizer

        return self.x_t

def generate_simplex(dim):
    p = np.random.uniform(0,100, size=[dim])
    return p/np.sum(p)

def l2_loss(x,y):
    return np.linalg.norm(x-y)

def grad_l2_loss(x,y):
    return x - y

if __name__ == "__main__":
    dim = 5          # dimension
    eta = 0.9                   # stepsize
    T = 500                     # number of steps
    losses = np.zeros([T,1])    # loss values placeholder
    threshold = 0.0001          # convergence threshold

    p = generate_simplex(dim)
    print("objective simplex: ", p)

    online_md = omd(dim, eta)
    # determine initial value
    x_t = online_md.x_t
    print("initialization: ", x_t)

    for t in range(0,T):
        x_t = online_md.step(grad_l2_loss(x_t, p))
        loss_t = l2_loss(x_t,p)
        losses[t] = loss_t

        # check for convergence
        if np.abs(x_t - p).all() < threshold:
            print("solution converged at iter ", t)
            break
    
    plt.plot(losses)
    plt.ylabel("loss")
    plt.xlabel("iter")
    plt.savefig("plots/losses.png")

    print("learned simplex: ", x_t)

