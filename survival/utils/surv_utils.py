"""
This contains survival utils to compute p-value, ci, etc

from https://github.com/huangzhii/SALMON
"""
import numpy as np
import lifelines
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazards = hazards.cpu().numpy().reshape(-1)
    labels = labels.data.cpu().numpy()

    label = []
    hazard = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            
    label = np.asarray(label)
    
    median = np.median(hazard)
    hazards_dichotomize = np.zeros([len(hazard)], dtype=int)
    hazards_dichotomize[hazard > median] = 1
    if hazards_dichotomize.shape != label.shape:
        label = label.reshape(hazards_dichotomize.shape)
    correct = np.sum(hazards_dichotomize == label)
    return correct / len(label)


def cox_log_rank(hazards, labels, survtime_all):
    # hazardsdata = hazards.cpu().numpy().reshape(-1)
    # labels = labels.data.cpu().numpy().reshape(-1)
    # survtime_all = survtime_all.data.cpu().numpy().reshape(-1)

    labels = labels.data.cpu().numpy().reshape(-1)
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    # import pdb
    # pdb.set_trace()
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    # import pdb
    # pdb.set_trace()
    label = np.asarray(label)
    hazard = np.asarray(hazard)
    surv_time = np.asarray(surv_time)

    median = np.median(hazard)
    hazards_dichotomize = np.zeros([len(hazard)], dtype=int)
    hazards_dichotomize[hazard > median] = 1
    idx = hazards_dichotomize == 0
    T1 = surv_time[idx]
    T2 = surv_time[~idx]
    E1 = label[idx]
    E2 = label[~idx]
    # try:
    #     results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    # except:
    #     import pdb; pdb.set_trace()
    if len(T1) == 0:
        return -1
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]:
                        concord = concord + 1
                    elif hazards[j] < hazards[i]:
                        concord = concord + 0.5

    return (concord / total)


def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy().reshape(-1)
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.cpu().numpy().reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]) and not np.isnan(survtime_all[i]) and not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)
    # try:
    #     concordance_index(new_surv, -new_hazard, new_label)
    # except:
    #     import pdb; pdb.set_trace()
    return (concordance_index(new_surv, -new_hazard, new_label))


def frobenius_norm_loss(a, b):
    loss = torch.sqrt(torch.sum(torch.abs(a - b)**2))
    return loss
