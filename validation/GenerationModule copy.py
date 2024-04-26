import torch
import torch.nn.functional as F


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
    return F

def collect(Fs, Fi, Fd, w_sp, w_ip, w_dp):
    Fp = w_sp * Fs + w_ip * Fi + w_dp * Fd
    return Fp

def impulse(v_rec, t_rec, s_rec, v, theta, dt, tau, el, i, t):
    s = v > theta
    s_float = s.float()
    v = s_float * el + (1 - s_float) * (v - dt / tau * (v - el - i))
    v_rec = torch.cat((v_rec, v.unsqueeze(0)))
    t_rec = torch.cat((t_rec, torch.tensor([t])))
    s_rec = torch.cat((s_rec, s_float.unsqueeze(0)))   
    return s, v, v_rec, t_rec, s_rec

def perturbation(F_ext1, weight1, F_ext2, weight2, F_array, obs):
    F = F_array[0, obs] + weight1 * F_ext1 + weight2 * F_ext2
    idx = torch.argmin(torch.abs(F_array - F), dim=0)
    return idx, F

def step(likelihood, prior, obs, F_ext1, w1, F_ext2, w2):
    F = inference(likelihood, prior, obs)
    F2 = perturbation(F_ext1, w1, F_ext2, w2, F, obs)
    return F2

def generation(params):
    
    threshold, tau, w_sd, w_si, w_sp, w_id, w_ip, w_dp, decays1, decays2, decayi1, decayi2, decayd1, decayd2 = params
    sizes, sizei, sized = torch.tensor([50, 50, 50])
    mismatchs, mismatchi, mismatchd = torch.tensor([5, 5, 5])

    likelihood_s = create_likelihood_matrix(sizes, decays1)
    likelihood_i = create_likelihood_matrix(sizei, decayi1)
    likelihood_d = create_likelihood_matrix(sized, decayd1)
    prior_s = get_prior(30, sizes, decays2)
    prior_i = get_prior(30, sizei, decayi2)
    prior_d = get_prior(30, sized, decayd2)
    obs_s = torch.tensor((50 - mismatchs).int())
    obs_i = torch.tensor((50 - mismatchi).int())
    obs_d = torch.tensor((50 - mismatchd).int())

    v_rec = torch.empty(0); t_rec = torch.empty(0); s_rec = torch.empty(0)
    
    Fs = inference(likelihood_s, prior_s, obs_s)
    Fss = Fs[0, obs_s]
    Fi = inference(likelihood_i, prior_i, obs_i)
    Fii = Fi[0, obs_i]
    Fd = inference(likelihood_d, prior_d, obs_d)
    Fdd = Fd[0, obs_d]
    v = el = torch.tensor(0)
    dt = 0.1
    Fp = collect(Fss, Fii, Fdd, w_sp, w_ip, w_dp)

    for t in range(int(1/dt)):
        s, v, v_rec, t_rec, s_rec = impulse(v_rec, t_rec, s_rec, v, threshold, dt, tau, el, Fp, t)

    for _ in range(20):
        Fss = step(likelihood_s, prior_s, obs_s, Fii, w_si, Fdd, w_sd)
        Fii = step(likelihood_i, prior_i, obs_i, Fss, w_si, Fdd, w_id)
        Fdd = step(likelihood_d, prior_d, obs_d, Fss, w_sd, Fii, w_id)
        Fp = collect(Fss, Fii, Fdd, w_sp, w_ip, w_dp)

        for t in range(int(1/dt)):
            s, v, v_rec, t_rec, s_rec = impulse(v_rec, t_rec, s_rec, v, threshold, dt, tau, el, Fp, t)

    prediction = (s_rec.any() * 1).float()
    return prediction

