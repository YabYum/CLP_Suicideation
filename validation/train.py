import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def generation(params, Tstep):
    params = [torch.tensor(p, requires_grad=True) if not isinstance(p, torch.Tensor) else p for p in params]
    threshold, tau, w_sd, w_si, w_sp, w_id, w_ip, w_dp, decays1, decays2, decayi1, decayi2, decayd1, decayd2 = params
    threshold = threshold * 50; tau = tau * 50; decays1 = decays1 * 5; decays2 = decays2 * 5; decayi1 = decayi1 * 5; decayi2 = decayi2 * 5; decayd1 = decayd1 * 5; decayd2 = decayd2 * 5
    sizes, sizei, sized = torch.tensor([50, 50, 50]); obs_s, obs_i, obs_d = torch.tensor(23).int(), torch.tensor(25).int(), torch.tensor(25).int(); v, Fss, Fii, Fdd, dt = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0), torch.tensor(0.1)
    likelihood_s, likelihood_i, likelihood_d = create_likelihood_matrix(sizes, decays1), create_likelihood_matrix(sizei, decayi1), create_likelihood_matrix(sized, decayd1); prior_s, prior_i, prior_d = get_prior(25, sizes, decays2), get_prior(25, sizei, decayi2), get_prior(25, sized, decayd2)

    for _ in range(Tstep):
        Fss = step(likelihood_s, prior_s, obs_s, Fii, w_si, Fdd, w_sd)
        Fii = step(likelihood_i, prior_i, obs_i, Fss, w_si, Fdd, w_id)
        Fdd = step(likelihood_d, prior_d, obs_d, Fss, w_sd, Fii, w_id)
        Fp = collect(Fss, Fii, Fdd, w_sp, w_ip, w_dp)

        for t in range(int(1/dt)):
            v = impulse(v, dt, tau, Fp)
    
    pred = torch.sigmoid(v-threshold)
        
    return pred


def data_preparation(path):

    excel_file = path
    pd_data = pd.read_excel(excel_file, sheet_name='PDs')
    suicide_data = pd.read_excel(excel_file, sheet_name='Suicide')

    assert pd_data['serial'].equals(suicide_data['serial'])

    X = pd_data.iloc[:, 1:26]
    y = suicide_data['SI'] 

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print("Train set:", X_train.shape, y_train.shape, 
          "Test set:", X_test.shape, y_test.shape, 
          "Validation set:", X_val.shape, y_val.shape)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    return X, y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor

class SuicideRiskMLP(nn.Module):
    def __init__(self):
        super(SuicideRiskMLP, self).__init__()
        self.fc1 = nn.Linear(25, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 14)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def accuracy_test(model, xtest, ytest, Tstep):
    model.eval()
    TP = 0  # True Positives
    TN = 0  # True Negatives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    total = 0

    with torch.no_grad():
        for i in range(len(xtest)):  # Using len(xtest) to ensure all test samples are considered
            input = xtest[i]
            label = ytest[i]
            params = model(input)
            pred_result = generation(params, Tstep).squeeze(-1)
            pred = (pred_result > 0.5).float()           
            if pred == 1 and label == 1: # True positive
                TP += 1
            elif pred == 1 and label == 0: # False positive
                FP += 1
            elif pred == 0 and label == 1: # False negative
                FN += 1
            elif pred == 0 and label == 0: # True nagative
                TN += 1
            
            total += 1
            
    acc = (TP + TN) / total
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    false_positive_rate = FP / (FP + TN) if FP + TN != 0 else 0
    false_negative_rate = FN / (FN + TP) if FN + TP != 0 else 0

    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"False Negative Rate: {false_negative_rate:.4f}")

    return acc
    
def train_step(model, optimizer, criterion, sample, epochs, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, version, Tstep):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(sample):
            optimizer.zero_grad()
            input = X_train_tensor[i]
            label = y_train_tensor[i]
            params = model(input)
            pred_result = generation(params, Tstep).squeeze(-1)
            loss = criterion(pred_result, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        print("model", version, "epoch: ", epoch, " loss: ", total_loss)
    
    acc = accuracy_test(model, X_test_tensor, y_test_tensor, Tstep)
    print("Accuracy of current model on test dataset: ", acc)
    return acc

def copymlp(model):
    model_new = copy.deepcopy(model)
    criterion_new = nn.BCELoss()
    optimizer_new = optim.Adam(model_new.parameters(),lr = 0.001)
    return model_new, criterion_new, optimizer_new


def iteration_train(nums_iteration, subsequent_samples, subsequent_epochs, xtrain, ytrain, xtest, ytest, SuicideRiskMLP, T):
    model = SuicideRiskMLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    try:
        acc = train_step(model, optimizer, criterion, 50, 50, xtrain, ytrain, xtest, ytest, version=0, Tstep=T)

    except Exception as e:
        print(f"An error occurred during initial training: {e}")
        return None, None

    model_backup = copy.deepcopy(model)
    acc_backup = acc

    for iteration in range(nums_iteration):
        try:

            model_backup = copy.deepcopy(model)
            acc_backup = acc
            
            model, criterion, optimizer = copymlp(model)
            acc = train_step(model, optimizer, criterion, subsequent_samples, subsequent_epochs, xtrain, ytrain, xtest, ytest, version=(iteration + 1), Tstep=T)
            
            if acc > 0.73:
                break
        except Exception as e:

            print(f"An error occurred during iteration {iteration + 1}: {e}")
            return model_backup, acc_backup

    return model, acc


def draw_roc(model, X_val_tensor, y_val_tensor, T):

    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for i in range(280):
            input = X_val_tensor[i]
            label = y_val_tensor[i].item()
            params = model(input)
            pred_result = generation(params, Tstep=T).squeeze(-1)
            predictions.append(pred_result.item())
            labels.append(label)
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def generation_prog(params, Tstep, mismatch_s, mismatch_i, mismatch_r):
    params = [torch.tensor(p, requires_grad=True) if not isinstance(p, torch.Tensor) else p for p in params]
    threshold, tau, w_sd, w_si, w_sp, w_id, w_ip, w_dp, decays1, decays2, decayi1, decayi2, decayd1, decayd2 = params
    threshold = threshold * 50; tau = tau * 50; decays1 = decays1 * 5; decays2 = decays2 * 5; decayi1 = decayi1 * 5; decayi2 = decayi2 * 5; decayd1 = decayd1 * 5; decayd2 = decayd2 * 5
    sizes, sizei, sized = torch.tensor([50, 50, 50]); obs_s, obs_i, obs_d = torch.tensor(25 - mismatch_s).int(), torch.tensor(25 - mismatch_i).int(), torch.tensor(25 - mismatch_r).int(); v, Fss, Fii, Fdd, dt = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0), torch.tensor(0.1)
    likelihood_s, likelihood_i, likelihood_d = create_likelihood_matrix(sizes, decays1), create_likelihood_matrix(sizei, decayi1), create_likelihood_matrix(sized, decayd1); prior_s, prior_i, prior_d = get_prior(25, sizes, decays2), get_prior(25, sizei, decayi2), get_prior(25, sized, decayd2)

    for _ in range(Tstep):
        Fss = step(likelihood_s, prior_s, obs_s, Fii, w_si, Fdd, w_sd)
        Fii = step(likelihood_i, prior_i, obs_i, Fss, w_si, Fdd, w_id)
        Fdd = step(likelihood_d, prior_d, obs_d, Fss, w_sd, Fii, w_id)
        Fp = collect(Fss, Fii, Fdd, w_sp, w_ip, w_dp)

        for t in range(int(1/dt)):
            v = impulse(v, dt, tau, Fp)
    
    pred = torch.sigmoid(v-threshold)
        
    return pred

def estimate_bear_ability(model, X, Tstep, sample):
    bear_ability = []
    for i in range(sample):
        params = model(X[i])
        mismatch_s, mismatch_i, mismatch_r = 0, 0, 0

        for j in range(50):
            mismatch_s = j 
            bear = generation_prog(params, Tstep, mismatch_s, 0, 0)
            if bear > 0.5: break

        for k in range(50):
            mismatch_i = k 
            bear = generation_prog(params, Tstep, 0, mismatch_i, 0)
            if bear > 0.5: break

        for l in range(50):
            mismatch_r = l 
            bear = generation_prog(params, Tstep, 0, 0, mismatch_r)
            if bear > 0.5: break
    
        bear_point = [mismatch_s, mismatch_i, mismatch_r]
        bear_ability.append(bear_point)
    return bear_ability

def visualize_bear_ability(bear_ability, y):
    points_red = [bear_ability[i] for i in range(len(bear_ability)) if y[i] == 1]
    points_black = [bear_ability[i] for i in range(len(bear_ability)) if y[i] == 0]
    x_red, y_red, z_red = zip(*points_red); x_black, y_black, z_black = zip(*points_black)
    jitter_amount = 0.20    
    x_red_jittered = np.array(x_red) + np.random.normal(0, jitter_amount, len(x_red))
    y_red_jittered = np.array(y_red) + np.random.normal(0, jitter_amount, len(y_red))
    z_red_jittered = np.array(z_red) + np.random.normal(0, jitter_amount, len(z_red))
    x_black_jittered = np.array(x_black) + np.random.normal(0, jitter_amount, len(x_black))
    y_black_jittered = np.array(y_black) + np.random.normal(0, jitter_amount, len(y_black))
    z_black_jittered = np.array(z_black) + np.random.normal(0, jitter_amount, len(z_black))

    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(x_red_jittered, y_red_jittered, z_red_jittered, c='red', s=20, alpha=0.7)
    ax.scatter(x_black_jittered, y_black_jittered, z_black_jittered, c='black', s=20, alpha=0.7)

    ax.set_xlabel('Symbolic suicide', fontsize=11)
    ax.set_ylabel('Imaginary suicide', fontsize=11)
    ax.set_zlabel('Desire of death', fontsize=11)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


    plt.legend()
    plt.show()