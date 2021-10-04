import jax
import jax.numpy as jnp

def euclidean_distance(point_a, point_b):
    return jnp.sqrt(((point_a - point_b)**2).sum(axis=-1))
@jax.jit
def brute_force_single(point_a, points_b, sq_distance_centers, min_distance, max_distance, bin_width):
    
    
    distance = ((point_a - points_b)**2).sum(axis=-1)
    mask = jnp.logical_and((distance < max_distance), (distance > min_distance))
    return jnp.where(mask[:,None], 
                    jax.nn.softmax(- (distance[:,None] - sq_distance_centers[None,:])**2 / bin_width[None,:], axis=1), 
                    jnp.broadcast_to(0., (points_b.shape[0], sq_distance_centers.shape[0]))).sum(axis=0)
@jax.jit
def brute_force_loop(points_a, points_b, sq_distance_centers, min_distance, max_distance, bin_width):
    pair_counts = jnp.zeros_like(sq_distance_centers)
    carry = (pair_counts, points_b, sq_distance_centers, min_distance, max_distance, bin_width)
    def scan_func(carry, x):
        pair_counts, points_b, sq_distance_centers, min_distance, max_distance, bin_width = carry
        new_pair_counts = brute_force_single(x, points_b, sq_distance_centers, min_distance, max_distance, bin_width)
        pair_counts = jax.ops.index_add(pair_counts, jax.ops.index[:], new_pair_counts)
        return (pair_counts, points_b, sq_distance_centers, min_distance, max_distance, bin_width), 0
    
    carry, _ = jax.lax.scan(scan_func, carry, points_a)
    pair_counts, points_b, sq_distance_centers, min_distance, max_distance, bin_width = carry
    return pair_counts

@jax.jit
def make_kd_tree(points, dim, i=0):
    if points.shape[0] > 1:
        sorter = jnp.argsort(points[:,i])
        points = points[sorter]
        i = (i + 1) % dim
        half = points.shape[0] >> 1
        return [
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half]
        ]
    elif points.shape[0] == 1:
        return [None, None, points[0]]

