import pandas as pd
import numpy as np
import snp_network
import os
import torch
from snp_input import get_data, get_pretrain_dataset
from torch import nn
from net_test_summary import summarise_net
from snp_network import get_net_savename
from pandas_plink import read_plink1_bin
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Number of GPUs: ", torch.cuda.device_count())

def get_or_train_net(parameters, phenos):
    pretrain_snv_toks = get_pretrain_dataset(phenos, parameters)
    net_file = parameters["saved_nets_dir"] + get_net_savename(parameters)
    if os.path.exists(net_file):
        print("reloading file: {}".format(net_file))
        net = snp_network.get_transformer(parameters, pretrain_snv_toks)
        # net = nn.DataParallel(net, [0])
        net.load_state_dict(torch.load(net_file))
        # net = net.to(0)
    else:
        print("Training new net, no saved net in file '{}'".format(net_file))
        net = snp_network.train_everything(parameters)
        if torch.cuda.device_count()>1:
            torch.save(net.module.state_dict(), net_file)
        else:
            torch.save(net.state_dict(), net_file)
            
    return net, pretrain_snv_toks


def run():
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
    parameters['use_pcs']           = False    
    parameters['num_heads']         = 1
    parameters['num_layers']        = 1
    parameters['batch_size']        = 32
    parameters['num_epochs']        = 50
    parameters['test_only']         = False
    parameters['pretrain_epochs']   = 5
    parameters['num_phenos']        = 0
    parameters['num_pcs']           = 0
    parameters['pheno_dim']         = 0
    parameters['embed_dim']         = 36
    parameters['linformer_k']       = 36
    parameters['use_linformer']     = True
    parameters['input_filtering']   = 'random_test_verify'
    
    parameters['gwas_dir']          = "../data/"    
    parameters['cache_dir']         = f"../cache/"
    parameters['saved_nets_dir']    = f"../ablation/model/{parameters['num_heads']}heads_{parameters['num_layers']}layers/"
    parameters['label_dir']         = "../data/genotypeTransformer_matched_age_hare_gender_smoker.csv"
    
    if not os.path.exists(parameters['saved_nets_dir']):
        os.makedirs(parameters['saved_nets_dir'])
    if not os.path.exists(parameters['cache_dir']):
        os.makedirs(parameters['cache_dir'])
    if not os.path.exists(parameters["saved_nets_dir"] + "verify/"):
        os.makedirs(parameters["saved_nets_dir"] + "verify/")
    if not os.path.exists(parameters["saved_nets_dir"] + "test/"):
        os.makedirs(parameters["saved_nets_dir"] + "test/")
        
    df_label = pd.read_csv(parameters['label_dir'])
    print(df_label.shape)
    assert df_label.shape[0]==df_label['MVPCore_id'].nunique()
    print(df_label['case'].value_counts())

    bed_file = parameters['gwas_dir']+f"{parameters['pretrain_base']}.bed"
    bim_file = parameters['gwas_dir']+f"{parameters['pretrain_base']}.bim"
    fam_file = parameters['gwas_dir']+f"{parameters['pretrain_base']}.fam"
    
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    df_label = df_label[df_label['MVPCore_id'].isin(geno_tmp['sample'].values)]
    print(df_label['case'].value_counts())

    df_label["eid"] = df_label["MVPCore_id"].apply(str)
    
    cancerdx_0 = df_label[df_label['case']==0]
    cancerdx_1 = df_label[df_label['case']==1]
    
    pheno = pd.concat([cancerdx_0, cancerdx_1])

    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno = get_data(pheno, parameters)

    net_file = parameters["saved_nets_dir"] + get_net_savename(parameters)
    net, pretrain_snv_toks = get_or_train_net(parameters, pheno)

    print("summarising test-set results")
    save_path_verify = parameters["saved_nets_dir"] + "verify/" + get_net_savename(parameters)
    summarise_net(net, verify, parameters, save_path_verify, geno)
    save_path_test   = parameters["saved_nets_dir"] + "test/" + get_net_savename(parameters)
    summarise_net(net, test, parameters, save_path_test, geno)


if __name__ == "__main__":
    run()
