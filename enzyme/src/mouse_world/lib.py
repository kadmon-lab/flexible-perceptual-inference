import jax
import jax.numpy as jnp
def p_end(counter, a):
    """
    Leave probability uniform in [0, a]
    """
    counter, a = jnp.array(counter), jnp.array(a)
    p = jax.lax.cond(counter <= 0, 
                    lambda: 0., 
                    lambda: jax.lax.cond(counter == a,   
                            lambda: 1.,
                            lambda: 1. / (a - counter)))
    return p