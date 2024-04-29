import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def generate_digital(personality):
    facet_scores = {}
    for facet in personality["Personality Trait Facet"]:
        for key, value in facet.items():
            facet_scores[value] = round(random.uniform(0, 3), 3)

    return facet_scores

def generate_event(life_events):
    event_scores = {}
    for event in life_events['events']:
        for key, value in event.items():
            if key != 'impact_score':
                event_name = value
                impact_score = event['impact_score'] 
                random_score = random.randint(0, impact_score)
                event_scores[event_name] = random_score
    return event_scores

def visualize_scale(scalescores, figsize, fontsize, title):
    facet_names = list(scalescores.keys())[::-1]
    scores = list(scalescores.values())[::-1]
    plt.figure(figsize=figsize)
    plt.barh(facet_names, scores, color='skyblue')
    plt.xticks(fontsize=12); plt.yticks(fontsize=fontsize)  
    plt.xlabel('Scores')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def model_params(pid):
    threshold = (pid['Hostility'] / 5) * 50
    tau = ((5 + pid['Restricted Affectivity'] - pid['Emotional Liability']) / 10) * 20
    w_sp = w_ip = w_dp = (pid['Impulsivity'] + pid['Anhedonia'] - pid['Restricted Affectivity'] + 5) / 15
    w_si = (pid['Attention Seeking'] + pid['Submissiveness']) / 10
    decays1 = ((pid['Eccentricity'] + pid['Unusual Beliefs & Experiences']) / 10) * 5
    decays2 = ((pid['Depressivity'] + pid['Submissiveness'] - pid['Grandiosity'] + 5 + pid['Anxiousness']) /20) * 5
    decayi1 = ((pid['Perceptual Dysregulation'] + pid['Callousness']) / 10) * 5
    decayi2 = ((10 - pid['Manipulativeness'] - pid['Rigid Perfectionism'] + pid['Intimacy Avoidance'] + pid['Separation Insecurity'] + pid['Withdrawl']) / 25) *5
    decayd1, decayd2 = 2, 1
    w_sd, w_id = [0.1, 0.2]
    return threshold, tau, w_sd, w_si, w_sp, w_id, w_ip, w_dp, decays1, decays2, decayi1, decayi2, decayd1, decayd2

def mismatch(srrs):
    symbolic_events = srrs["Jail Term"] + srrs["Fired at work"] + srrs["Retirement"] + srrs["Business readjustment"] + srrs["Change in financial state"] + \
                      srrs["Change to a different line of work"] + srrs["Mortgage over $20,000"] + srrs["Foreclosure of mortgage or loan"] + srrs["Mortgage or loan less than $20,000"] + \
                      srrs["Change in responsibilities at work"] + srrs["Trouble with in-laws"] + srrs["Begin or end school"] + srrs["Change in living conditions"] + \
                      srrs["Change in work hours or conditions"] + srrs["Change in residence"] + srrs["Minor violation of the law"] + srrs["Change in schools"] + \
                      srrs["Change in recreations"] + srrs["Change in church activities"]

    imaginary_events = srrs["Divorce"] + srrs["Marital Separation"] + srrs["Sex difficulties"] + \
                       srrs["Change in number of arguments with spouse"]  + srrs["Trouble with boss"] + \
                       srrs["Revisions of personal habits"] + srrs["Change in number of family get-togethers"
                                                               ] + srrs["Son or daughter leaving home"]

    real_events = srrs["Death of spouse"] + srrs["Death of a close friend"] + srrs["Death of close family member"] + srrs["Pregnancy"] + \
                   srrs["Personal injury or illness"] + srrs["Change in health of family member"]

    mismatchs, mismatchi, mismatchd = 10 * symbolic_events/583, 10 * imaginary_events/303, 10 * real_events/337
    return mismatchs, mismatchi, mismatchd

def create_likelihood_matrix(size, ambiguity):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distance = abs(i - j)
            matrix[i, j] = np.exp(-0.5 * (distance / ambiguity) ** 2)
    norm_matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return norm_matrix

def log_stable(x, minval=1e-30):
    return np.log(np.maximum(x, minval))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def infer_states(observation_index, likelihood_matrix, prior):
    log_likelihood = log_stable(likelihood_matrix[observation_index, :])
    log_prior = log_stable(prior)
    qs = softmax(log_likelihood + log_prior)
    return qs

def kl_divergence(p, q):
    return (log_stable(p) - log_stable(q)).dot(p)

def get_prior(prior_state, size, ambiguity):
    prior = np.zeros(size)
    for i in range(size):
        distance = abs(i - prior_state)
        prior[i] = np.exp(-0.5 * (distance / ambiguity) ** 2)
    norm_prior = prior / prior.sum()
    return norm_prior

def plot_likelihood(matrix):
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Likelihood Matrix')
    plt.xlabel('Observations')
    plt.ylabel('States')
    plt.show()

def plot_belief(belief, title):
    plt.figure(figsize=(5, 2))
    plt.plot(belief, marker = None, linestyle='-', color='black')
    plt.title(f'{title}')
    plt.xlabel('State')
    plt.grid(True)
    plt.show()

def inference(likelihood, prior, obs):
    qs = infer_states(obs, likelihood, prior)
    dkl = kl_divergence(qs, likelihood[obs, :])
    evidence = log_stable(prior)
    F = dkl - evidence
    return qs, F

def perturbation(F_ext1, weight1, F_ext2, weight2, F_array, obs):
    F = F_array[obs] + (weight1 * F_ext1) + (weight2 * F_ext2)
    idx = (np.abs(F_array - F)).argmin()
    return idx, F

def step(likelihood, prior, obs, F_ext1, w1, F_ext2, w2):
    qs, F = inference(likelihood, prior, obs)
    qs2, F2 = perturbation(F_ext1, w1, F_ext2, w2, F, obs)
    return qs, F, qs2, F2

def collect(Fs, Fi, Fd, w_sp, w_ip, w_dp):
    Fp = w_sp * Fs + w_ip * Fi + w_dp * Fd
    return Fp

def impulse(v_rec, t_rec, s_rec, v, theta, dt, tau, el,i, t):
    s = v > theta
    v = s* el + (1-s) * (v - dt/ tau * ((v - el) - i))
    v_rec = np.append(v_rec, v)
    t_rec = np.append(t_rec, t)
    s_rec = np.append(s_rec, s)
    return s,v,v_rec,t_rec,s_rec

def plot_sims(s_rec, v_rec, Fss_history, Fii_history, Fdd_history, Fp_history, Fs_min_history, Fi_min_history, Fd_min_history):
    Fi_min_history1 = [x + 0.6 for x in Fi_min_history]
    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 3)

    ax1 = plt.subplot(gs[2])
    ax1.plot(s_rec, '.', markersize=20, color='red')
    ax1.axis([0, 210, 0.8, 1.2])
    ax1.set_xticks([])
    ax1.set_ylabel("Impulses")

    ax2 = plt.subplot(gs[1])
    ax2.plot(v_rec, linestyle='-.')
    ax2.set_xticks([])
    ax2.set_ylabel("Accumulation")

    ax3 = plt.subplot(gs[0])
    ax3.plot(Fss_history, linestyle='-', color='black')
    ax3.plot(Fii_history, linestyle='-', color='green')
    ax3.plot(Fdd_history, linestyle='-', color='gray')
    ax3.plot(Fp_history, linestyle='-', color='red')
    ax3.plot(Fs_min_history, linestyle='--', color='black')
    ax3.plot(Fi_min_history1, linestyle='--', color='green')
    ax3.plot(Fd_min_history, linestyle='--', color='gray')
    ax3.legend()
    ax3.set_ylabel('Free energy')

    plt.tight_layout()
    plt.show()

def generation(threshold, tau, w_sd, w_si, w_sp, w_id, w_ip, w_dp, decays1, decays2, decayi1, decayi2, decayd1, decayd2, mismatchs, mismatchi, mismatchd):
    sizes, sizei, sized = [50, 50, 50]
    likelihood_s, likelihood_i, likelihood_d = create_likelihood_matrix(sizes, decays1), create_likelihood_matrix(sizei, decayi1), create_likelihood_matrix(sized, decayd1)
    prior_s, prior_i, prior_d = get_prior(49, sizes, decays2), get_prior(49, sizei, decayi2), get_prior(49, sized, decayd2)
    obs_s, obs_i, obs_d = [int(49-mismatchs),int(49-mismatchi), int(49-mismatchd)]
    dt, el = 0.1, 0
    Fss_history, Fii_history, Fdd_history, Fp_history = [], [], [], []
    Fs_min_history, Fi_min_history, Fd_min_history = [], [], []
    qss_history, qsi_history, qsd_history = [], [], []
    v_rec, t_rec, s_rec = np.array([]), np.array([]), np.array([])

    Fss_history, Fii_history, Fdd_history, Fp_history = [], [], [], []
    Fs_min_history, Fi_min_history, Fd_min_history = [], [], []
    qss_history, qsi_history, qsd_history = [], [], []
    v_rec, t_rec, s_rec = np.array([]), np.array([]), np.array([])

    qss, Fs = inference(likelihood_s, prior_s, obs_s)
    Fss = Fs[obs_s]
    Fss_history.append(Fss)
    Fs_min = Fs[Fs.argmin()]
    Fs_min_history.append(Fs_min)
    qss_history.append(qss.argmax())

    qsi, Fi = inference(likelihood_i, prior_i, obs_i)
    qsi_history.append(qsi.argmax())
    Fii = Fi[obs_i]
    Fii_history.append(Fii)
    Fi_min = Fi[Fi.argmin()]
    Fi_min_history.append(Fi_min)

    qsd, Fd = inference(likelihood_d, prior_d, obs_d)
    qsd_history.append(qsd.argmax())
    Fdd = Fd[obs_d]
    Fdd_history.append(Fdd)
    Fd_min = Fd[Fd.argmin()]
    Fd_min_history.append(Fd_min)

    Fp = collect(Fss, Fii, Fdd, w_sp, w_ip, w_dp)
    Fp_history.append(Fp)

    v = el
    for t in range (int(1/dt)):
        s, v, v_rec, t_rec, s_rec = impulse(v_rec, t_rec, s_rec, v, threshold, dt, tau, el, Fp, t)

    for _ in range(20):
        qss, Fs, qsss, Fss = step(likelihood_s, prior_s, obs_s, Fii, w_si, Fdd, w_sd)
        Fss_history.append(Fss)
        Fs_min_history.append(Fs.min())
        qss_history.append(qsss)
        qsi, Fi, qsii, Fii = step(likelihood_i, prior_i, obs_i, Fss, w_si, Fdd, w_id)
        Fii_history.append(Fii)
        Fi_min_history.append(Fi.min())
        qsi_history.append(qsii)

        qsd, Fd, qsdd, Fdd = step(likelihood_d, prior_d, obs_d, Fss, w_sd, Fii, w_id)
        Fdd_history.append(Fdd)
        Fd_min_history.append(Fd.min())
        qsd_history.append(qsdd)
        Fp = collect(Fss, Fii, Fdd, w_sp, w_ip, w_dp)
        Fp_history.append(Fp)

        for t in range (int(1/dt)):
            s, v, v_rec, t_rec, s_rec = impulse(v_rec, t_rec, s_rec, v, threshold, dt, tau, el, Fp, t)
    return s_rec, v_rec, Fss_history, Fii_history, Fdd_history, Fp_history, Fs_min_history, Fi_min_history, Fd_min_history
