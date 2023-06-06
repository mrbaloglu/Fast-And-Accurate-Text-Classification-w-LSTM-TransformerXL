import time
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Multinomial, Categorical
import numpy as np
import pandas as pd
import pickle
from typing import Dict
from tqdm import tqdm

# set up parameters
CHUNCK_SIZE = 20
BATCH_SIZE = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''Compute FLOPs(Floating point operations) of the models.
Specify the computational costs for the models above

cnn_cost: CNN model(which is separated from CNN_LSTM above)
s_cost: policy s(stopping module)
c_cost: policy c(classifier)
lstm_cost: LSTM model(which is separated from CNN_LSTM above)
cnn_whole: CNN model with whole reading(400 words).

The costs below are based on the size of one chunk(20 words).
'''
cnn_cost = 1024000
s_cost = 50050
c_cost = 16770 
n_cost = 50310
lstm_cost = 286720
clstm_cost = cnn_cost + lstm_cost
cnn_whole = 25344000


def sample_policy_s(ht, policy_s):
    '''
    Draw a stopping decision from a Bernoulli distribution specified by policy s.
    '''
    s_prob = policy_s(ht)
    m = Bernoulli(s_prob)
    stop_decision = m.sample()
    # compute the log prob
    log_prob_s = m.log_prob(stop_decision)
    return stop_decision, log_prob_s

def sample_policy_c(output_c):
    '''
    Draw a label from a multinomial distribution specified by policy c.
    '''
    prob_c = F.softmax(output_c, dim=1)
    m = Categorical(prob_c)
    pred_label = m.sample()
    log_prob_c = m.log_prob(pred_label)
    return pred_label, log_prob_c

def sample_policy_n(ht, policy_n):
    '''
    Draw an action from a multinomial distribution specified by policy n.
    '''
    action_probs = policy_n(ht)
    m = Categorical(action_probs)
    step = m.sample()
    log_prob_n = m.log_prob(step)
    return step.item(), log_prob_n
    
def compute_policy_value_losses(cost_ep, loss, saved_log_probs, baseline_value_ep, alpha, gamma):
    '''compute the policy losses and value losses for the current episode
    '''
    # normalise cost
    norm_cost_ep = (cost_ep - np.mean(cost_ep)) / (np.std(cost_ep) + 1e-7)
    #print('norm_cost_ep:', norm_cost_ep)
    reward_ep = - alpha * norm_cost_ep
    reward_ep[-1] -= loss.item()
    # compute discounted rewards
    discounted_rewards = [r * gamma ** i for i, r in enumerate(reward_ep)]
    policy_loss_ep = []
    value_losses = []
    for i, log_prob in enumerate(saved_log_probs):
        # baseline_value_ep[i].item(): updating the policy loss doesn't include the gradient of baseline values
        advantage = sum(discounted_rewards) - baseline_value_ep[i].item()
        policy_loss_ep.append(log_prob * advantage)
        value_losses.append((sum(discounted_rewards) - baseline_value_ep[i]) ** 2)   
    return policy_loss_ep, value_losses


def evaluate(clstm, policy_s, policy_n, policy_c, loader, max_steps: int = 5):
    '''
    Evaluate a model with skimming, rereading, and early stopping
    and compute the average FLOPs per data.
    '''
    # set the models in evaluation mode
    clstm.eval()
    policy_s.eval()
    policy_n.eval()
    policy_c.eval()
    count_all = 0
    count_correct = 0
    start = time.time()
    # the sum of FLOPs of the iterator set
    flops_sum = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating..."):
            label = y.to(device).long() # for cross entropy loss, the long type is required
            text = x.to(device).view(CHUNCK_SIZE, BATCH_SIZE, CHUNCK_SIZE) # transform 1*400 to 20*1*20
            curr_step = 0
            n_rnn_layers = clstm.n_rnn_layers
            lstm_hidden_dim = clstm.lstm_hidden_dim
            h_0 = torch.zeros([n_rnn_layers,1,lstm_hidden_dim]).to(device)
            c_0 = torch.zeros([n_rnn_layers,1,lstm_hidden_dim]).to(device)
            count = 0
            while curr_step < 20 and count < max_steps: # loop until a text can be classified or currstep is up to 20
                count += 1
                # pass the input through cnn-lstm and policy s
                text_input = text[curr_step] # text_input 1*20
                ht, ct = clstm(text_input, h_0, c_0)  # 1 * 128
                h_0 = ht #.unsqueeze(0) # NUM_RNN_LAYERS * 1 * LSTM_HIDDEN_DIM, next input of lstm
                c_0 = ct
                ht = ht[-1, :, :]# ht.view(1, ht.shape[0] * ht.shape[2])
                # draw a stop decision
                stop_decision, log_prob_s = sample_policy_s(ht, policy_s)
                flops_sum += clstm_cost + s_cost
                stop_decision = stop_decision.item()
                if stop_decision == 1: # classify
                    break
                else:
                    # draw an action (reread or skip)
                    step, log_prob_n = sample_policy_n(ht, policy_n)
                    flops_sum += n_cost
                    curr_step += int(step)  # reread or skip
            # draw a predicted label
            output_c = policy_c(ht)
            flops_sum += c_cost
            # draw a predicted label 
            pred_label, log_prob_c = sample_policy_c(output_c)
            if pred_label.item() == label:
                count_correct += 1
            count_all += 1
    print('Evaluation time elapsed: %.2f s' % (time.time() - start))
    avg_flop_per_sample = int(flops_sum / len(loader))
    print('Average FLOPs per sample: ', avg_flop_per_sample)
    return count_all, count_correct


def evaluate_lm(clstm, policy_s, policy_n, policy_c, loader, max_steps: int = 5):
    '''
    Evaluate a model with skimming, rereading, and early stopping
    and compute the average FLOPs per data.
    '''
    # set the models in evaluation mode
    clstm.eval()
    policy_s.eval()
    policy_n.eval()
    policy_c.eval()
    count_all = 0
    count_correct = 0
    start = time.time()
    # the sum of FLOPs of the iterator set
    flops_sum = 0
    with torch.no_grad():
        for x, xmask, y in tqdm(loader, desc="Evaluating..."):
            label = y.to(device).long() # for cross entropy loss, the long type is required
            text = x.to(device).view(CHUNCK_SIZE, BATCH_SIZE, CHUNCK_SIZE) # transform 1*400 to 20*1*20
            text_mask = xmask.to(device).view(CHUNCK_SIZE, BATCH_SIZE, CHUNCK_SIZE)
            curr_step = 0
            n_rnn_layers = clstm.n_rnn_layers
            lstm_hidden_dim = clstm.lstm_hidden_dim
            h_0 = torch.zeros([n_rnn_layers,1,lstm_hidden_dim]).to(device)
            c_0 = torch.zeros([n_rnn_layers,1,lstm_hidden_dim]).to(device)
            count = 0
            while curr_step < 20 and count < max_steps: # loop until a text can be classified or currstep is up to 20
                count += 1
                # pass the input through cnn-lstm and policy s
                text_input = text[curr_step] # text_input 1*20
                text_mask_input = text_mask[curr_step]
                ht, ct = clstm(text_input, text_mask_input, h_0, c_0)  # 1 * 128
                h_0 = ht #.unsqueeze(0) # NUM_RNN_LAYERS * 1 * LSTM_HIDDEN_DIM, next input of lstm
                c_0 = ct
                ht = ht[-1, :, :]# ht.view(1, ht.shape[0] * ht.shape[2])
                # draw a stop decision
                stop_decision, log_prob_s = sample_policy_s(ht, policy_s)
                flops_sum += clstm_cost + s_cost
                stop_decision = stop_decision.item()
                if stop_decision == 1: # classify
                    break
                else:
                    # draw an action (reread or skip)
                    step, log_prob_n = sample_policy_n(ht, policy_n)
                    flops_sum += n_cost
                    curr_step += int(step)  # reread or skip
            # draw a predicted label
            output_c = policy_c(ht)
            flops_sum += c_cost
            # draw a predicted label 
            pred_label, log_prob_c = sample_policy_c(output_c)
            if pred_label.item() == label:
                count_correct += 1
            count_all += 1
    print('Evaluation time elapsed: %.2f s' % (time.time() - start))
    avg_flop_per_sample = int(flops_sum / len(loader))
    print('Average FLOPs per sample: ', avg_flop_per_sample)
    return count_all, count_correct


def evaluate_transformer(transformer, policy_s, policy_n, policy_c, loader, max_steps: int = 5):
    '''
    Evaluate a model with skimming, rereading, and early stopping
    and compute the average FLOPs per data.
    '''
    # set the models in evaluation mode
    transformer_config = transformer.config
    memory_length = transformer_config["memory_length"]
    num_blocks = transformer_config["num_blocks"]
    embed_dim = transformer_config["embed_dim"]
    trns_input_dim = transformer_config["trns_input_dim"]
    max_episode_length = 20
    num_workers = 1
    transformer.eval()
    policy_s.eval()
    policy_n.eval()
    policy_c.eval()
    count_all = 0
    count_correct = 0
    start = time.time()
    # the sum of FLOPs of the iterator set
    flops_sum = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating..."):
            label = y.to(device).long() # for cross entropy loss, the long type is required
            text = x.to(device).view(CHUNCK_SIZE, BATCH_SIZE, CHUNCK_SIZE) # transform 1*400 to 20*1*20
            curr_step = 0
            # Setup placeholders for each worker's current episodic memory
            memory = torch.zeros((num_workers, memory_length, num_blocks, trns_input_dim), dtype=torch.float32).to(device)
            memory_mask = torch.tril(torch.ones((num_workers, memory_length)), diagonal=-1).to(device)
            repetitions = torch.repeat_interleave(torch.arange(0, memory_length).unsqueeze(0), memory_length - 1, dim = 0).long()
            memory_indices = torch.stack([torch.arange(i, i + memory_length) for i in range(max_episode_length - memory_length + 1)]).long()
            memory_indices = torch.cat((repetitions, memory_indices)).to(device)
            count = 0
            while curr_step < 20 and count < max_steps: # loop until a text can be classified or currstep is up to 20
                count += 1
                # pass the input through cnn-lstm and policy s
                text_input = text[curr_step] # text_input 1*20
                ht, memory_t = transformer(text_input, memory, memory_mask, memory_indices)
                # ht_ = ht.view(1, ht.shape[0] * ht.shape[2])
                # draw a stop decision
                stop_decision, log_prob_s = sample_policy_s(ht, policy_s)
                flops_sum += clstm_cost + s_cost
                stop_decision = stop_decision.item()
                if stop_decision == 1: # classify
                    break
                else:
                    # draw an action (reread or skip)
                    step, log_prob_n = sample_policy_n(ht, policy_n)
                    flops_sum += n_cost
                    curr_step += int(step)  # reread or skip
            # draw a predicted label
            output_c = policy_c(ht)
            flops_sum += c_cost
            # draw a predicted label 
            pred_label, log_prob_c = sample_policy_c(output_c)
            if pred_label.item() == label:
                count_correct += 1
            count_all += 1
    print('Evaluation time elapsed: %.2f s' % (time.time() - start))
    avg_flop_per_sample = int(flops_sum / len(loader))
    print('Average FLOPs per sample: ', avg_flop_per_sample)
    return count_all, count_correct

def openDfFromPickle(path: str) -> pd.DataFrame:
    """Open stored dataframes from pickle files.

    Args:
        path (str): File path.

    Returns:
        pd.DataFrame: Dataframe read.
    """
    assert path.endswith(".pkl"), \
        f"Given file path must end with .pkl, got {path}."

    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def calculate_stats_from_cm(confusion_matrix: np.ndarray, macro_avg: bool = True) -> Dict[str, float]:
    """Calculate accuracy, precision, recall and F1 Score from a given confusion matrix.
       By default, macro avg. of precision, recall and F1 Score is calculated.
    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        macro_avg (bool): Whether to calculate macro avg. or not. Set it false to calculate micro avg.
    Returns:
        Dict[str, float]: Dictionary of stats -- {"accuracy": acc_val, "precision": prec_val, "recall": rec_val, "f1": f1_val}
    """
    if macro_avg:
        precisions = np.zeros((confusion_matrix.shape[0],))  # per class precision scores
        recalls = np.zeros((confusion_matrix.shape[0],)) # per class recall scores
        f1s = np.zeros((confusion_matrix.shape[0],)) # per class f1 scores
        for j in range(confusion_matrix.shape[0]):
            if np.sum(confusion_matrix[j, :]) != 0:
                precisions[j] = confusion_matrix[j, j] / np.sum(confusion_matrix[j, :])
                
            if np.sum(confusion_matrix[:, j]) != 0:
                recalls[j] = confusion_matrix[j, j] / np.sum(confusion_matrix[:, j])
            
            if precisions[j] + recalls[j] != 0:
                f1s[j] = 2*precisions[j]*recalls[j] / (precisions[j]+recalls[j])

            
        macro_f1 = np.average(f1s) # if sum(f1s) > 0 else 0.
        macro_precision = np.average(precisions) # if sum(precisions) > 0 else 0.
        macro_recall = np.average(recalls) # if sum(recalls) > 0 else 0.
        accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix) # if np.sum(confusion_matrix) > 0 else 0.

        return {"accuracy": accuracy, "precision": macro_precision, "recall": macro_recall, "f1": macro_f1}
    else: 
        raise NotImplementedError


def evaluate_earlystop(clstm, policy_s, policy_c, loader):
    '''
    Evaluate a early stopping model with only a stopping module
    and compute the average FLOPs per data.
    '''
    # set the models in evaluation mode
    clstm.eval()
    policy_s.eval()
    policy_c.eval()
    count_all = 0
    count_correct = 0
    start = time.time()
    # the sum of FLOPs of the iterator set
    flops_sum = 0
    with torch.no_grad():
        for batch in loader:
            label = batch.label.to(torch.long) # 64
            text = batch.text.view(CHUNCK_SIZE, BATCH_SIZE, CHUNCK_SIZE) # transform 1*400 to 20*1*20
            curr_step = 0
            # set up the initial input for lstm
            h_0 = torch.zeros([1,1,128]).to(device)
            c_0 = torch.zeros([1,1,128]).to(device)
            saved_log_probs = []
            while (curr_step < 20):
                '''
                loop until stop decision equals 1 
                or the whole text has been read
                '''
                # read a chunk
                text_input = text[curr_step]
                # hidden state
                ht, ct = clstm(text_input, h_0, c_0)  # 1 * 128
                h_0 = ht.unsqueeze(0).cuda()  # 1 * 1 * 128, next input of lstm
                c_0 = ct
                # draw a stop decision
                stop_decision, log_prob_s = sample_policy_s(ht, policy_s)
                stop_decision = stop_decision.item()
                flops_sum += clstm_cost + s_cost
                if stop_decision == 1:
                    break
                else:
                    curr_step += 1
            # output of classifier       
            output_c = policy_c(ht)
            flops_sum += c_cost
            # draw a predicted label 
            pred_label, log_prob_c = sample_policy_c(output_c)
            if pred_label.item() == label:
                count_correct += 1
            count_all += 1     
    print('Evaluation time elapsed: %.2f s' % (time.time() - start))
    avg_flop_per_sample = int(flops_sum / len(loader))
    print('Average FLOPs per sample: ', avg_flop_per_sample)  
    return count_all, count_correct


def print_model_parm_flops(model, input):
    '''
    Compute FLOPs of a model.
    '''
    multiply_adds = False
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        print('input size', input[0].size())
        print('output size:', output[0].size())
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)-1
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)

    list_linear=[] 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
        
    list_sig = []
    def sig_hook(self, input, output):
        list_sig.append(input[0].nelement())
    
    list_softmax = []
    def softmax_hook(self, input, output):
        print(input[0].nelement())
        list_softmax.append(input[0].nelement())
        
    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
        list_pooling.append(flops)
    
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Sigmoid):
                net.register_forward_hook(sig_hook)
            if isinstance(net, torch.nn.Softmax):
                net.register_forward_hook(softmax_hook)
            return
        for c in childrens:
                foo(c)
    foo(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = model(input.to(device))
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_sig) +sum(list_softmax)) 
    return total_flops


