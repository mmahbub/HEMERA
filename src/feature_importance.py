import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import captum
from captum.attr import IntegratedGradients, LayerIntegratedGradients, TokenReferenceBase, visualization

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import torch
import time
from torch import nn, tensor
from torch.utils import data
from pandas_plink import read_plink1_bin
from os.path import exists

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import snp_network
from snp_input import get_data, get_pretrain_dataset
from snp_network import transformer_from_encoder, get_encoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def preprocess_pheno(parameters, label_dir):
    df_label = pd.read_csv(label_dir)
    print(df_label['case'].value_counts())
        
    bed_file = parameters['gwas_dir']+parameters['plink_base']+".bed"
    bim_file = parameters['gwas_dir']+parameters['plink_base']+".bim"
    fam_file = parameters['gwas_dir']+parameters['plink_base']+".fam"

    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    df_label = df_label[df_label['MVPCore_id'].isin(geno_tmp['sample'].values)]
    df_label["eid"] = df_label["MVPCore_id"].apply(str)
    cancerdx_0 = df_label[df_label['case']==0]
    cancerdx_1 = df_label[df_label['case']==1]
    pheno = pd.concat([cancerdx_0, cancerdx_1])

    return pheno


def get_or_train_net(parameters, phenos):
    pretrain_snv_toks = get_pretrain_dataset(phenos, parameters)
    net_file = parameters["saved_nets_dir"] + get_net_savename(parameters)
    print('net_file: ', net_file)
    if os.path.exists(net_file):
        print("reloading file: {}".format(net_file))
        net = snp_network.get_transformer(parameters, pretrain_snv_toks)
        # net = nn.DataParallel(net, [0])
        net.load_state_dict(torch.load(net_file))
        net = net.to(0)
    else:
        print("Error")
    return net, pretrain_snv_toks


def get_net_savename(parameters: dict):
    parameters = {k:parameters[k] for k in ["num_heads", "num_layers", "batch_size", "num_epochs", "pretrain_epochs"]}
    par_str = "_".join("{}{}".format(k,v) for k, v in parameters.items() if k in ["num_heads", "num_layers", "batch_size", "num_epochs", "pretrain_epochs"])
    par_str += '.net'
    return par_str.replace('/', '-')
    
def get_full_param_string(params: dict):
    str = "\n".join("{}: {}".format(k, v) for k, v in params.items())
    return str

def get_avg_reference(net, test_data, parameters, geno):
    # save parameters
    param_string = get_full_param_string(parameters)
    batch_size = 64
    test_iter = data.DataLoader(test_data, (int)(batch_size), shuffle=False)
    test_loss = 0.0
    loss = nn.CrossEntropyLoss()
    device = 0
    net = net.cuda(device)
    net.eval()
    
    embeddings = []
    with torch.no_grad():
        for pcs, _, _, _, _, pos, tX, tY in tqdm(test_iter):
            # phenos = phenos.to(device)
            tX = tX.to(device)
            emb = torch.mean(net.encoder(tX), dim=0)
            embeddings.append(emb.cpu())
        
    return embeddings

def get_testdata(test):
    batch_size = 1
    test_iter = data.DataLoader(test, (int)(batch_size), shuffle=False)
    xList, pcsList = [], []
    for pcs, _, _, _, _, _, x, _ in tqdm(test_iter):
        xList.append(x)
        pcsList.append(pcs)
    return xList, pcsList

def get_attributes(test, reference_indices, lig):
    """
    Get the attributes for all samples.
    """
    import gc
    xList, pcsList = get_testdata(test)

    attrDict = {}
    for i, x in tqdm(enumerate(xList[:])):
        x = x.to(0)
        pcs = pcsList[i]
        with torch.no_grad():
            attr, delta = lig.attribute(inputs=(x,pcs),
                                        baselines=(reference_indices,pcs),
                                        n_steps=5,
                                        target=1,
                                        return_convergence_delta=True)

        attrDict[tuple(x.to('cpu'))] = attr.to('cpu')
        del x
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
    return attrDict


def captum_featureimp(net, test, parameters, train_embeddings, ATTR_SAVE_PATH):
    
    def forward_with_sigmoid(input, pcs):
        output = net(input, pcs)
        return output#[...,1]

    # seq_len = parameters['encoder_size'] 
    # token_reference = TokenReferenceBase(reference_token_idx=0) #nan token
    # reference_indices = token_reference.generate_reference(seq_len, device=device).unsqueeze(0)

    reference_indices = torch.stack(train_embeddings).mean(dim=0).mean(dim=1)[1:].unsqueeze(0)
    
    net.eval()
    torch.no_grad()
    # ig = IntegratedGradients(forward_with_sigmoid)
    lig = LayerIntegratedGradients(forward_with_sigmoid, net.encoder.embedding)
    attrDict = get_attributes(test, reference_indices, lig)
    return attrDict

def calculate_average_importance(net, test, attrDict, ATTRAVG_SAVE_PATH, ATTRVAR_SAVE_PATH):    

    attrDict_avg = {k:np.array(torch.mean(v.squeeze(0)[1:],dim=1)) for k,v in tqdm(attrDict.items())}
    
    averages_dict = {}
    for xList,aList in tqdm(attrDict_avg.items()):
        xList = np.array(xList[0])
        for pos, (val1,val2) in enumerate(zip(xList,aList)):
            key = (val1,pos)
            if key in averages_dict:
                averages_dict[key].append(val2)
            else:
                averages_dict[key] = [val2]
    
    averages = {key:np.mean(values) for key,values in tqdm(averages_dict.items())}
    variance = {key:np.var(values,ddof=1) for key,values in tqdm(averages_dict.items())}
    
    # patient_ct_per_snp_pos = []
    # for key,values in averages_dict.items():
    #     patient_ct_per_snp_pos.append(len(values))    
    # weighted_averages = {key:(sum(values)/len(values))*(len(values)/sum(patient_ct_per_snp_pos))
                         # for key,values in averages_dict.items()}
    
    return averages, variance

def get_finaldf(parameters, avg, pheno, test_ids): 
    print("using data from:", parameters['gwas_dir'])
    bed_file = parameters['gwas_dir']+parameters['plink_base']+".bed"
    bim_file = parameters['gwas_dir']+parameters['plink_base']+".bim"
    fam_file = parameters['gwas_dir']+parameters['plink_base']+".fam"
    
    data_raw   = read_plink1_bin(bed_file, bim_file, fam_file)
    usable_ids = pheno["eid"].values.tolist()
    data_raw   = data_raw[data_raw["sample"].isin(usable_ids)]
    
    x = data_raw[data_raw['sample'].isin([test_ids[0]])].to_dataframe().reset_index()

    x['refined_pos'] = range(len(x))

    mapping_df = x[['refined_pos', 'chrom', 'snp', 'pos', 'a0', 'a1', 'genotype']]
    data_df = pd.DataFrame([(key[0],key[1],value) for key,value in tqdm(avg.items())],
                           columns=['snp','refined_pos','attr'])
    merged_df = pd.merge(data_df,mapping_df,on='refined_pos')
    merged_df.sort_values(by=['chrom','refined_pos'],inplace=True)
    
    chromosome_offsets = merged_df.groupby('chrom')['refined_pos'].max().sort_values().cumsum()
    chromosome_midpoints = (chromosome_offsets.diff().fillna(chromosome_offsets.iloc[0])/2)
    chromosome_midpoints = [x+chromosome_midpoints[i-1]  if i>0 else x for i,x in enumerate(chromosome_midpoints)]
    
    chromosomes = sorted(merged_df['chrom'].apply(int).unique())
    chromosomes = [str(c) for c in chromosomes]

    return merged_df, chromosome_midpoints, chromosomes

def main(which_head, which_layer, num_epoch):
    fold = 1
    
    parameters = snp_network.default_parameters
    parameters['pretrain_base']     = 'data' #filtered data
    parameters['plink_base']        = 'data' #filtered data
    parameters['continue_training'] = False
    parameters['train_new_encoder'] = False
    parameters['test_frac']         = 0.20
    parameters['verify_frac']       = 0.10
    parameters['lr']                = 1e-7
    parameters['pt_lr']             = 1e-7
    parameters['encoder_size']      = 378866
    parameters['use_phenos']        = False
    parameters['output_type']       = 'tok'
    parameters['num_heads']         = which_head
    parameters['num_layers']        = which_layer
    parameters['num_epochs']        = num_epoch
    parameters['batch_size']        = 32 
    parameters['pretrain_epochs']   = 5
    parameters['num_phenos']        = 0
    parameters['test_only']         = False
    parameters['use_pcs']            = False
    parameters['num_pcs']           = 0
    parameters['embed_dim']         = 36
    parameters['linformer_k']       = 36
    parameters['use_linformer']     = True
    parameters['input_filtering']   = 'random_test_verify'
    
    parameters['gwas_dir']          = "../data/"    
    parameters['label_dir']         = "../data/genotypeTransformer_matched_age_hare_gender_smoker.csv"
    parameters['cache_dir']         = f"../cache_fold{fold}/"
    parameters['saved_nets_dir']    = f"../ablation/model_fold{fold}/{parameters['num_heads']}heads_{parameters['num_layers']}layers/"

    pheno = preprocess_pheno(parameters, parameters['label_dir'])
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno = get_data(pheno, parameters)
    
    net_file = parameters["saved_nets_dir"] + get_net_savename(parameters)
    net, pretrain_snv_toks = get_or_train_net(parameters, pheno)

    # print("summarising test-set results")
    # test_res = summarise_net(net, test, parameters, geno, EUR_ONLY=EUR_ONLY)
    # print(metrics.precision_score(test_res['Actual'],test_res['Predicted']), metrics.recall_score(test_res['Actual'],test_res['Predicted']))

    SAVE_PATH = parameters['saved_nets_dir'] + 'captum_num_epochs_MeanRef50' + str(parameters['num_epochs']) + '/'
    print(SAVE_PATH)
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    ATTR_SAVE_PATH = SAVE_PATH + "attribute_scores.pkl"
    print(ATTR_SAVE_PATH)

    ATTRAVG_SAVE_PATH = SAVE_PATH + "attribute_by_(snp,pos).pkl"
    print(ATTRAVG_SAVE_PATH)

    ATTRVAR_SAVE_PATH = SAVE_PATH + "attrVar_by_(snp,pos).pkl"
    print(ATTRVAR_SAVE_PATH)

    TRAIN_EMBEDDING_PATH = SAVE_PATH + "train_embeddings.pkl"
    print(TRAIN_EMBEDDING_PATH)

    print("Extracting/loading training embeddings . . . ")
    if not os.path.exists(TRAIN_EMBEDDING_PATH):
        train_embeddings = get_avg_reference(net, train, parameters, geno)
        pickle.dump(train_embeddings, open(TRAIN_EMBEDDING_PATH, "wb"))
    else:
        train_embeddings = pickle.load(open(TRAIN_EMBEDDING_PATH, "rb"))
    
    print("Calculating/loading feature importance . . . ")    
    if not os.path.exists(ATTR_SAVE_PATH):
        attrDict = captum_featureimp(net, test, parameters, train_embeddings, ATTR_SAVE_PATH)
        pickle.dump(attrDict, open(ATTR_SAVE_PATH, "wb"))
    else:
        start_time = time.time()
        attrDict = pickle.load(open(ATTR_SAVE_PATH, "rb"))
        end_time = time.time()
        print(f"Loading attributions took {(end_time-start_time)/60:.6f} minutes.")

    print("Calculating/loading average feature importance . . . ")    
    if not os.path.exists(ATTRAVG_SAVE_PATH):
        averages, variance = calculate_average_importance(net, test, attrDict, ATTRAVG_SAVE_PATH, ATTRVAR_SAVE_PATH)
        pickle.dump(averages, open(ATTRAVG_SAVE_PATH,"wb"))
        pickle.dump(variance, open(ATTRVAR_SAVE_PATH,"wb"))
    else:
        start_time = time.time()
        averages = pickle.load(open(ATTRAVG_SAVE_PATH,"rb"))
        end_time = time.time()
        print(f"Loading avg. attributions took {(end_time-start_time)/60:.6f} minutes.")

    print("Saving final dataframe . . . ")    
    merged_df, chromosome_midpoints, chromosomes = get_finaldf(parameters, averages, pheno, test_ids)
    merged_df.to_csv(SAVE_PATH+'merged_df.csv',index=False)
    pickle.dump(chromosome_midpoints, open(SAVE_PATH+'chromosome_midpoints.pkl','wb'))
    pickle.dump(chromosomes, open(SAVE_PATH+'chromosomes.pkl','wb'))


which_head = 1
which_layer = 1
num_epoch = 50
main(which_head, which_layer, num_epoch)