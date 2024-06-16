from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as F

from visualization import states_to_imgs


EPSILON = 1e-8

def sample_mi(oh_features, var_type='discrete'):
    """ Estimate the mutual information between all pairwise sets of features.
    
    Args:
        oh_features: torch.Tensor, shape (n_samples, n_features, feature_dim)
        var_type: str, 'discrete' or 'continuous'
    
    Returns:
        torch.Tensor, shape ()
    """
    assert var_type in ['discrete', 'continuous'], \
        f'var_type must be either discrete or continuous, not {var_type}!'

    if var_type == 'continuous':
        raise NotImplementedError('Continuous MI is not implemented yet!')

    feature_entropies = one_hot_entropy(oh_features.mean(dim=0))

    n_samples, n_features = oh_features.shape[:2]
    total_mutual_info = torch.tensor(0, dtype=torch.float32, device=oh_features.device)
    total_su = torch.tensor(0, dtype=torch.float32, device=oh_features.device)
    for i in range(n_features):
        for j in range(i+1, n_features):
            oh_xs = oh_features[:, i]
            oh_ys = oh_features[:, j]

            # H(x|y) = -sum_x[ sum_y[ p(x, y) log(p(x, y) / p(y)) ] ]
            joint_probs = torch.mm(oh_xs.T, oh_ys) / n_samples
            y_probs = torch.mean(oh_ys, dim=0).unsqueeze(0)
            conditional_entropy_mat = joint_probs * torch.log(
                (joint_probs / (y_probs + EPSILON)) + EPSILON)
            conditional_entropy = -torch.sum(conditional_entropy_mat)

            mutual_info = feature_entropies[i] - conditional_entropy # I(x, y) = H(x) - H(x|y)
            total_mutual_info += mutual_info
            total_su += 2 * mutual_info / (
                feature_entropies[i] + feature_entropies[j] + EPSILON)

    return total_mutual_info, total_su

class HashableTensorWrapper(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor, *args, hash=None, **kwargs):
        return super().__new__(cls, tensor, *args, **kwargs)

    def __init__(self, tensor, hash=None):
        self._tensor = tensor
        self._hash = hash

    # def __getattr__(self, attr):
    #     return getattr(self._tensor, attr)
    def __getitem__(self, i):
        return HashableTensorWrapper(self._tensor.__getitem__(i))
    def __hash__(self):
        if self._hash is None:
            # print('Warning: HashableTensorWrapper hash not set!')
            self._hash = hash_tensor(self._tensor).item()
        return self._hash
    def set_hash(self, hash):
        self._hash = hash
    def __repr__(self):
        return self._tensor.__repr__()
    def __str__(self):
        return self._tensor.__str__()
    def __len__(self):
        return self._tensor.__len__()
    def __eq__(self, other):
        return (self._tensor == other._tensor).all()

def as_hashable(x):
    if isinstance(x, torch.Tensor):
        return HashableTensorWrapper(x)
    raise NotImplementedError(f'Cannot convert {type(x)} to hashable!')

def hash_tensor(tensor, bins=int(1e6)):
    order_add = torch.linspace(0, 1, tensor.shape.numel(), device=tensor.device)
    order_add = order_add.reshape(tensor.shape)
    return torch.quantize_per_tensor(int(1e6) * (tensor + order_add), 0.1, 0, torch.qint32) \
        .int_repr().reshape(-1).sum() % bins

def hash_tensors(tensors, bins=int(1e6)):
    order_add = torch.linspace(0, 1, tensors[0].shape.numel(), device=tensors[0].device)
    order_add = order_add.reshape([1] + list(tensors[0].shape))
    return (torch.quantize_per_tensor(int(1e6) * (tensors + order_add), 0.1, 0, torch.qint32) \
        .int_repr().reshape(tensors.shape[0], -1).sum(-1) % bins).tolist()

def to_hashable_tensor_list(tensor):
    """ Convert a N-D tensor into a list of (N-1)-D hashable tensors. """
    hashes = hash_tensors(tensor)
    tensor_list = []
    for i in range(len(tensor)):
        tensor_list.append(HashableTensorWrapper(tensor[i], hash=hashes[i]))
    return tensor_list

def categorical_kl_div(pred, target):
    """ Calculate KL Divergence for categorical distributions. """
    return torch.sum(target * torch.log(target / (pred + EPSILON) + EPSILON), dim=-1)

def one_hot_cross_entropy(pred, target):
    """ Calculate the cross entropy between two one-hot vectors. """
    return -torch.sum(target * torch.log(pred + EPSILON), dim=-1)

def one_hot_entropy(probs):
    """ Calculate the cross entropy between two one-hot vectors. """
    # Input: (n_features, discrete_dim)
    return -torch.sum(probs * torch.log(probs + EPSILON), dim=-1)

# dist = Categorical(torch.tensor([0.6, 0.4, 0.0]))
# logits = torch.rand((1, dist.support.upper_bound + 1,), requires_grad=True)
# optimizer = torch.optim.Adam((logits,), lr=1e-2)
# print(F.softmax(logits, dim=1))
# for i in range(10000):
#     samples = dist.sample((1000,))
#     oh_samples = F.one_hot(samples, num_classes=dist.support.upper_bound + 1).float()
#     probs = F.softmax(logits, dim=1)
#     # loss = one_hot_cross_entropy(oh_samples, probs).mean()
#     loss = categorical_kl_div(probs, oh_samples).mean()
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i % 1000 == 0:
#         print(F.softmax(logits, dim=1), loss.item())

# Reference: https://gnarlyware.com/blog/mutual-information-exaxmple-with-categorical-variables/
def sample_symmetric_uncertainty(features, var_type='discrete'):
    """ Estimate the symmetric uncertainty between all pairwise sets of features.
    
    Args:
        features: torch.Tensor, shape (n_samples, n_features)
        var_type: str, 'discrete' or 'continuous'
    
    Returns:
        torch.Tensor, shape (n_features, n_features)
    """
    assert var_type in ['discrete', 'continuous'], \
        f'var_type must be either discrete or continuous, not {var_type}!'

    if var_type == 'continuous':
        raise NotImplementedError('Continuous symmetric uncertainty is not implemented yet!')

    # Formula: 2 * I(x, y) / (H(x), H(y))
    # I(x, y) = H(x) - H(x|y) = H(y) - H(y|x) 
    discrete_dim = features.max() + 1
    one_hot_features = F.one_hot(features, num_classes=discrete_dim).float()
    feature_entropies = sample_entropy(features.transpose(0, 1), var_type=var_type)

    n_samples, n_features = features.shape
    sym_uncertainty = torch.eye(n_features, dtype=torch.float32, device=features.device)
    for i in range(features.shape[1]):
        for j in range(i+1, features.shape[1]):
            oh_xs = one_hot_features[:, i]
            oh_ys = one_hot_features[:, j]

            # H(x|y) = -sum_x[ sum_y[ p(x, y) log(p(x, y) / p(y)) ] ]
            joint_probs = torch.mm(oh_xs.T, oh_ys) / n_samples
            y_probs = torch.mean(oh_ys, dim=0).unsqueeze(0)
            conditional_entropy_mat = joint_probs * torch.log(
                (joint_probs / (y_probs + EPSILON)) + EPSILON)
            conditional_entropy = -torch.sum(conditional_entropy_mat)

            mutual_info = feature_entropies[i] - conditional_entropy # I(x, y) = H(x) - H(x|y)
            sym_uncertainty[i, j] = 2 * mutual_info / (feature_entropies[i] + feature_entropies[j])
            sym_uncertainty[j, i] = sym_uncertainty[i, j]

    return sym_uncertainty

def triu_avg(mat, nan_replacement=0):
    """ Average the upper triangular part of a matrix.
    
    Args:
        mat: torch.Tensor, square matrix
    """
    return mat.triu(1).nan_to_num(nan_replacement).sum() / \
        (np.prod(mat.shape) - mat.shape[0]) * 2

def sample_entropy(xs, var_type='discrete'):
    """ Estimate the entropy of the samples from random variable X.
    
    Args:
        xs: torch.Tensor, shape (n_features, n_samples)
    """
    assert var_type in ['discrete', 'continuous'], \
        f'var_type must be either discrete or continuous, not {var_type}!'

    if var_type == 'continuous':
        raise NotImplementedError('Continuous entropy is not implemented yet!')

    discrete_dim = xs.max() + 1
    # (n_features, n_smaples) -> (n_features, n_samples, discrete_dim)
    one_hots = F.one_hot(xs, num_classes=discrete_dim).float()
    # (n_features, n_samples, discrete_dim) -> (n_features, discrete_dim)
    probs = one_hots.mean(dim=1)
    # (n_features, discrete_dim) -> (n_features,)
    entropy = -torch.sum(probs * torch.log(probs + EPSILON), dim=1)
    return entropy

def obs_to_img(obs, cat=False, env_name=None, rev_transform=None):
    """ Convert a tensor observation into an image. """
    
    if len(obs.shape) in (1, 3):
        obs = obs.unsqueeze(0)
    obs = states_to_imgs(obs, env_name, transform=rev_transform)
    obs = torch.from_numpy(obs)

    assert len(obs.shape) == 4, 'Observations must be 3D or 4D'
    if cat:
        imgs = rearrange(obs, 'n c h w -> h (n w) c')
    else:
        imgs = rearrange(obs, 'n c h w -> n h w c')
    # Take last channel if not using RGB (or 3 channels in general)
    if imgs.shape[-1] != 3:
        imgs = imgs[..., -1]
    imgs = imgs.clip(0, 1).squeeze(0).cpu().numpy()
    return imgs

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)

#     def test_sym_uncertainty(xdist=None, ydist=None, xs=None, ys=None, n_samples=1000):
#         if xs is None:
#             xs = xdist.sample((n_samples,))
#         if ys is None:
#             ys = ydist.sample((n_samples,))
#         all_samples = torch.stack([xs, ys], dim=1)
#         sym_uncertainty = sample_symmetric_uncertainty(all_samples)
#         return sym_uncertainty

#     print('\n===== Test 1 =====')
#     dist1 = Categorical(torch.tensor([1.0, 0.0]))
#     dist2 = dist1
#     print(test_sym_uncertainty(dist1, dist2))

#     print('\n===== Test 2 =====')
#     dist1 = Categorical(torch.tensor([0.5, 0.5]))
#     dist2 = dist1
#     print(test_sym_uncertainty(dist1, dist2))

#     print('\n===== Test 3 =====')
#     xs = torch.randint(0, 5, (1000,))
#     print(test_sym_uncertainty(xs=xs, ys=xs))

#     print('\n===== Test 4 =====')
#     xs = torch.randint(0, 5, (1000000,))
#     n_share = 500000
#     ys = torch.randint(0, 5, (1000000-n_share,))
#     ys = torch.cat([xs[:n_share], ys])
#     print(test_sym_uncertainty(xs=xs, ys=ys))

#     print('\n===== Test 5 =====')
#     v1 = torch.randint(0, 16, (10000,))
#     v2 = torch.randint(0, 16, (10000,))
#     v3 = torch.randint(0, 4, (10000,))
#     others = [torch.zeros((10000,), dtype=torch.long) for _ in range(5)]
    
#     all_samples = torch.stack([v1, v2, v3, *others], dim=1)
#     sym_uncertainty = sample_symmetric_uncertainty(all_samples)
#     print(sym_uncertainty)
#     print('Sym Avg:', triu_avg(sym_uncertainty))

#     print('\n===== Test 6 =====')
#     v1 = v4 = v5 = torch.randint(0, 16, (10000,))
#     v2 = v6 = v7 = torch.randint(0, 16, (10000,))
#     v3 = v8 = torch.randint(0, 4, (10000,))
    
#     all_samples = torch.stack([v1, v2, v3, v4, v5, v6, v7, v8], dim=1)
#     sym_uncertainty = sample_symmetric_uncertainty(all_samples)
#     print(sym_uncertainty)
#     print('Sym Avg:', triu_avg(sym_uncertainty))