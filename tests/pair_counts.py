


import jax
import jax.numpy as jnp
from src.pair_counts import brute_force_loop, brute_force_single, make_kd_tree
import time
import proplot as pplt
import pandas as pd


n_part = 1000
box_size = 600.
key = jax.random.PRNGKey(42)




particles = pd.read_csv("data/CATALPTCICz0.466G960S1005638091_zspace.dat", usecols = (0,1,2), delim_whitespace=True, engine='c').values.astype(jnp.float32)
mask = ((particles < box_size) & (particles > 0)).all(axis=1)
particles = jax.device_put(particles[mask].copy())
n_part = particles.shape[0]
print(n_part, flush=True)
key, subkey = jax.random.split(key)
particles_rand = box_size * jax.random.uniform(subkey, (n_part, 3))
tree = make_kd_tree(particles, 3, i=0)

exit()
distance_edges = jnp.arange(50, 155, 5)
distance_centers = 0.5 * (distance_edges[1:] + distance_edges[:-1])
sq_distance_centers = distance_centers**2
sq_distance_edges = distance_edges**2
bin_width = jnp.diff(sq_distance_edges)

norm = n_part * (n_part - 1.)
norm_rand = particles_rand.shape[0] * (particles_rand.shape[0] - 1)
brute_force = jax.vmap(brute_force_single, in_axes=(0, None, None, None, None, None))

#s = time.time()
#jax_pair_counts = brute_force_loop(particles, particles, sq_distance_centers, sq_distance_edges[0], sq_distance_edges[-1], bin_width)
#print("JAX took", time.time() - s, "s", flush=True)
#print(jax_pair_counts/norm)

from Corrfunc.theory.DD import DD
s = time.time()
results = DD(1, 8, distance_edges,
                particles[:,0], particles[:,1], particles[:,2],
                boxsize=box_size, periodic=True)
#rr = DD(1, 8, distance_edges,
#                particles_rand[:,0], particles_rand[:,1], particles_rand[:,2],
#                boxsize=box_size, periodic=False)
print("Corrfunc took", time.time() - s, "s", flush=True)
print(results['npairs']/norm)
#rr = rr['npairs'] / norm_rand
#print((results['npairs'] - jax_pair_counts).mean())
#print(((results['npairs'] - jax_pair_counts)/norm).mean())

def rr_analytic(bin_low_bound, bin_high_bound, box_size):
    volume = 4 * jnp.pi * (bin_high_bound**3 - bin_low_bound**3) / 3
    normed_volume = volume / box_size **3
    return normed_volume
print(distance_centers)
rr = rr_analytic(distance_edges[:-1], distance_edges[1:], box_size)



fig, ax = pplt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
#ax[0].plot(distance_centers, (jax_pair_counts/norm/rr - 1))
ax[0].plot(distance_centers, distance_centers**2*(results['npairs']/norm/rr - 1), ls='--')


fig.savefig("plots/pair_counts.png", dpi=300)

