import torch

def create_likelihood_matrix(size, ambiguity):
    indices = torch.arange(size).unsqueeze(0)
    distance = torch.abs(indices - indices.T)
    matrix = torch.exp(-0.5 * (distance / ambiguity) ** 2)
    norm_matrix = matrix / matrix.sum(axis=1, keepdim=True)
    return norm_matrix

def get_prior(prior_state, size, ambiguity):
    indices = torch.arange(size)
    distance = torch.abs(indices - prior_state)
    prior = torch.exp(-0.5 * (distance / ambiguity) ** 2)
    norm_prior = prior / prior.sum()
    return norm_prior

def log_stable(x, minval=1e-30):
    return torch.log(torch.clamp(x, min=minval))


def infer_states(observation_index, likelihood_matrix, prior):
    log_likelihood = log_stable(likelihood_matrix[observation_index])
    log_prior = log_stable(prior)
    qs = torch.softmax(log_likelihood + log_prior)
    return qs

def inference(likelihood, prior, obs):
    qs = infer_states(obs, likelihood, prior)
    dkl = torch.kl_div(qs, likelihood[obs], reduction='batchmean')
    evidence = log_stable(prior).sum()
    F = dkl - evidence
    return qs, F

def perturbation(F_ext1, weight1, F_ext2, weight2, F_array, obs):
    F = F_array[obs] + (weight1 * F_ext1) + (weight2 * F_ext2)
    _, idx = torch.min(torch.abs(F_array - F), dim=0)
    return idx.item(), F

def step(likelihood, prior, obs, F_ext1, w1, F_ext2, w2):
    qs, F = inference(likelihood, prior, obs)
    idx, F2 = perturbation(F_ext1, w1, F_ext2, w2, F, obs)
    return F2

def collect(Fs, Fi, Fd, w_sp, w_ip, w_dp):
    Fp = w_sp * Fs + w_ip * Fi + w_dp * Fd
    return Fp

def impulse(v, dt, tau, i):
    return v - dt / tau * (v - i)

def infer_states(observation_indices, likelihood_matrix, prior):
    log_likelihood = log_stable(likelihood_matrix[observation_indices, :])
    log_prior = log_stable(prior).unsqueeze(0)
    qs = torch.softmax(log_likelihood + log_prior, dim=-1)
    return qs

def inference(likelihood, prior, obs):
    qs = infer_states(obs, likelihood, prior)
    dkl = torch.kl_div(qs, likelihood[obs, :])
    evidence = log_stable(prior)
    F = dkl - evidence
    return qs, F

def collect(Fs, Fi, Fd, w_sp, w_ip, w_dp):
    Fp = w_sp * Fs + w_ip * Fi + w_dp * Fd
    return Fp

def impulse(v, dt, tau, i):
    v = (v - dt / tau * (v - i))
    return v

def perturbation(F_ext1, weight1, F_ext2, weight2, F_array, obs):
    F = F_array[:, obs] + weight1 * F_ext1 + weight2 * F_ext2
    idx = torch.argmin(torch.abs(F_array - F), dim=0)
    return idx, F

def step(likelihood, prior, obs, F_ext1, w1, F_ext2, w2):
    qs, F = inference(likelihood, prior, obs)
    qs2, F2 = perturbation(F_ext1, w1, F_ext2, w2, F, obs)
    return F2

def generation(params):
    params = [torch.tensor(p, requires_grad=True) if not isinstance(p, torch.Tensor) else p for p in params]
    threshold, tau, w_sd, w_si, w_sp, w_id, w_ip, w_dp, decays1, decays2, decayi1, decayi2, decayd1, decayd2 = params
    threshold = threshold * 50; tau = tau * 50; decays1 = decays1 * 5; decays2 = decays2 * 5; decayi1 = decayi1 * 5; decayi2 = decayi2 * 5; decayd1 = decayd1 * 5; decayd2 = decayd2 * 5
    sizes, sizei, sized = torch.tensor([50, 50, 50]); obs_s, obs_i, obs_d = torch.tensor(20).int(), torch.tensor(25).int(), torch.tensor(25).int(); v, Fss, Fii, Fdd, dt = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0), torch.tensor(0.1)
    likelihood_s, likelihood_i, likelihood_d = create_likelihood_matrix(sizes, decays1), create_likelihood_matrix(sizei, decayi1), create_likelihood_matrix(sized, decayd1); prior_s, prior_i, prior_d = get_prior(25, sizes, decays2), get_prior(25, sizei, decayi2), get_prior(25, sized, decayd2)

    for _ in range(20):
        Fss = step(likelihood_s, prior_s, obs_s, Fii, w_si, Fdd, w_sd)
        Fii = step(likelihood_i, prior_i, obs_i, Fss, w_si, Fdd, w_id)
        Fdd = step(likelihood_d, prior_d, obs_d, Fss, w_sd, Fii, w_id)
        Fp = collect(Fss, Fii, Fdd, w_sp, w_ip, w_dp)

        for t in range(int(1/dt)):
            v = impulse(v, dt, tau, Fp)
    
    pred = torch.sigmoid(v-threshold)
        
    return pred