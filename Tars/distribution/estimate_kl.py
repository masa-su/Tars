from ..utils import  (
    gauss_unitgauss_kl,
    gauss_gauss_kl,
    epsilon
)

def kl_vs_prior(q, x, deterministic=False):
    if q.__class__.__name__=="Gaussian":
        mean, var = q.fprop(x, deterministic=deterministic)
        return gauss_unitgauss_kl(mean, var)

    elif q.__class__.__name__=="Bernoulli":
        mean = q.fprop(x, deterministic=deterministic)
        return mean * (T.log(mean + epsilon()) + T.log(2))
        
    else:
        raise Exception("You can't use this distribution as q")
        
def kl(q1, q2, x1, x2, deterministic=False):
    if q1.__class__.__name__=="Gaussian" and q2.__class__.__name__=="Gaussian":
        mean1, var1 = q1.fprop(x1, deterministic=deterministic)
        mean2, var2 = q2.fprop(x2, deterministic=deterministic)
        return gauss_gauss_kl(mean1, var1, mean2, var2)

    else:
        raise Exception("You can't use this distribution as q")

    
