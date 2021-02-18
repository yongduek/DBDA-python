
import jax.numpy as jnp
import jax.numpy as np    # it is the same anyways
from jax import random, vmap

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS


numpyro.set_host_device_count(4)  # MCMC chain

def mcmc_sample(data, num_samples=2000, num_chains=4, num_warmup=1000):
    # data = dict(W=W, L=L)
    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000, num_chains=4)
    mcmc.run(random.PRNGKey(0), **data)

    post = mcmc.get_samples()
    return post


class Sampler():
    def __init__(self, model, data=None):
        self.data = data
        self.num_warmup = 1000
        self.num_samples = 2000
        self.num_chains = 4
        self.mcmc = MCMC(NUTS(model), num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains)
        self.data = data
        
    def fit(self, data):
        self.data = data
        self.mcmc.run(random.PRNGKey(0), **data)
        self.post = self.mcmc.get_samples()
        return self.post  # posterior samples
    
    def predict(self, data):
        pass
#