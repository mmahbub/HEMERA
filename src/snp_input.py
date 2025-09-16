import pyximport
pyximport.install(setup_args={"script_args":["--verbose"]})

import os
import pandas
import numpy as np
from pandas_plink import read_plink1_bin
from pyarrow import parquet
import torch
from torch import tensor
from torch.utils import data
from process_snv_mat import get_tok_mat
import math
from os.path import exists
import pickle
import csv
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

def get_data(phenos, parameters):
    cache_dir = parameters['cache_dir']
    plink_base = parameters['plink_base']
    X_file = cache_dir + plink_base + '_X_cache.pickle'
    Y_file = cache_dir + plink_base + '_Y_cache.pickle'
    
    if exists(X_file) and exists(Y_file):
        with open(X_file, "rb") as f:
            X = pickle.load(f)
        with open(Y_file, "rb") as f:
            Y = pickle.load(f)
            print(Y_file, len(Y))
    else:
        print("reading data from plink")
        X, Y = read_from_plink(parameters, phenos, subsample_control=False)
        print("done, writing to pickle")
        with open(X_file, "wb") as f:
            pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
        with open(Y_file, "wb") as f:
            pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)
        print("done")

    if parameters['test_only']:
        test_ids, test = get_train_test_verify(X, Y, parameters)
        return test_ids, test, X, Y
    else:
        train_ids, train, test_ids, test, verify_ids, verify = get_train_test_verify(X, Y, parameters)
        return train_ids, train, test_ids, test, verify_ids, verify, X, Y


def ids_to_positions(id_list, ids_in_position):
    id_to_pos = {}
    for pos, id in enumerate(ids_in_position):
        id_to_pos[str(id)] = pos
    pos_list = [id_to_pos[i] for i in id_list]
    return np.array(pos_list)

def ids_to_pcs():
    content = open(parameters['pcs_file'],"r").readlines()
    content = [c.strip().split("\t") for c in tqdm(content)]
    ids_pcs = {c[0]:[float(x) for x in c[2:]] for c in tqdm(content[1:])}
    for v in tqdm(ids_pcs.values()): assert len(v)==30
    return ids_pcs  

def ids_to_age(parameters):
    df = pandas.read_csv(parameters['label_dir'] )
    ids_age = dict(zip(df['MVPCore_id'],df['age']))
    ids_age = {k:[v] for k,v in ids_age.items()}
    return ids_age  

def ids_to_gender(parameters):
    df = pandas.read_csv(parameters['label_dir'] )
    le  = LabelEncoder()
    df['gender_encoded'] = le.fit_transform(df['gender'])
    ids_gender = dict(zip(df['MVPCore_id'],df['gender_encoded']))
    ids_gender = {k:[v] for k,v in ids_gender.items()}
    return ids_gender  

def ids_to_smoker(parameters):
    df = pandas.read_csv(parameters['label_dir'] )
    le  = LabelEncoder()
    df['smoker_encoded'] = le.fit_transform(df['smoker'])
    ids_smoker = dict(zip(df['MVPCore_id'],df['smoker_encoded']))
    ids_smoker = {k:[v] for k,v in ids_smoker.items()}
    return ids_smoker

def ids_to_hare(parameters):
    df = pandas.read_csv(parameters['label_dir'] )
    le  = LabelEncoder()
    df['hare_encoded'] = le.fit_transform(df['hare'])
    ids_hare = dict(zip(df['MVPCore_id'],df['hare_encoded']))
    ids_hare = {k:[v] for k,v in ids_hare.items()}
    return ids_hare

    
def get_train_test_verify(geno, pheno, parameters):

    cancer = pheno["case"].values

    if parameters['test_only']:
        test_ids = get_train_test_verify_ids(pheno, parameters)
        test_all_pos = ids_to_positions(test_ids, geno.ids)
        
        print("Reading pcs file . . . ")
        pcs = ids_to_pcs()
        
        print("Creating test_pcs . . . ")
        test_pcs   = torch.tensor(np.array([pcs[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)

        print("Adding age . . . ")
        age = ids_to_age(parameters)
        print("Creating test_age . . . ")
        test_age   = torch.tensor(np.array([age[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)
        
        print("Adding gender . . . ")
        gender = ids_to_gender(parameters)
        print("Creating test_gender . . . ")
        test_gender   = torch.tensor(np.array([gender[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)
        
        print("Adding smoking status . . . ")
        smoker = ids_to_smoker(parameters)
        print("Creating test_smoker . . . ")
        test_smoker   = torch.tensor(np.array([smoker[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)

        print("Adding hare . . . ")
        hare = ids_to_hare(parameters)
        print("Creating test_hare . . . ")
        test_hare   = torch.tensor(np.array([hare[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)

        positions = geno.positions
        test_seqs = geno.tok_mat[test_all_pos,]
        test_phes = cancer[test_all_pos]
    
        device = "cuda" if torch.cuda.is_available() else "cpu"

        test_dataset = data.TensorDataset(
            # test_pheno_vec, 
            test_pcs,
            test_hare,
            test_age,
            test_gender,
            test_smoker,
            positions.repeat(len(test_seqs), 1),
            test_seqs, tensor(test_phes, dtype=torch.int64)
        )
        
        return test_ids, test_dataset

    else:
        train_ids, test_ids, verify_ids = get_train_test_verify_ids(pheno, parameters)
        
        train_all_pos = ids_to_positions(train_ids, geno.ids)
        test_all_pos = ids_to_positions(test_ids, geno.ids)
        verify_all_pos = ids_to_positions(verify_ids, geno.ids)
    
        print("Reading pcs file . . . ")
        pcs = ids_to_pcs()
        print("Creating train_pcs . . . ")
        train_pcs  = torch.tensor(np.array([pcs[id_] for id_ in tqdm(train_ids)]), dtype=torch.float32)
        print("Creating test_pcs . . . ")
        test_pcs   = torch.tensor(np.array([pcs[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)
        print("Creating verify_pcs . . . ")
        verify_pcs = torch.tensor(np.array([pcs[id_] for id_ in tqdm(verify_ids)]), dtype=torch.float32)
        
        print("Adding age . . . ")
        age = ids_to_age(parameters)
        print("Creating train_age . . . ")
        train_age  = torch.tensor(np.array([age[id_] for id_ in tqdm(train_ids)]), dtype=torch.float32)
        print("Creating test_age . . . ")
        test_age   = torch.tensor(np.array([age[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)
        print("Creating verify_age . . . ")
        verify_age = torch.tensor(np.array([age[id_] for id_ in tqdm(verify_ids)]), dtype=torch.float32)
        
        print("Adding gender . . . ")
        gender = ids_to_gender(parameters)
        print("Creating train_gender . . . ")
        train_gender  = torch.tensor(np.array([gender[id_] for id_ in tqdm(train_ids)]), dtype=torch.float32)
        print("Creating test_gender . . . ")
        test_gender   = torch.tensor(np.array([gender[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)
        print("Creating verify_gender . . . ")
        verify_gender = torch.tensor(np.array([gender[id_] for id_ in tqdm(verify_ids)]), dtype=torch.float32)
        
        print("Adding smoking status . . . ")
        smoker = ids_to_smoker(parameters)
        print("Creating train_smoker . . . ")
        train_smoker  = torch.tensor(np.array([smoker[id_] for id_ in tqdm(train_ids)]), dtype=torch.float32)
        print("Creating test_smoker . . . ")
        test_smoker   = torch.tensor(np.array([smoker[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)
        print("Creating verify_smoker . . . ")
        verify_smoker = torch.tensor(np.array([smoker[id_] for id_ in tqdm(verify_ids)]), dtype=torch.float32)
        
        print("Adding hare . . . ")
        hare = ids_to_hare(parameters)
        print("Creating train_hare . . . ")
        train_hare  = torch.tensor(np.array([hare[id_] for id_ in tqdm(train_ids)]), dtype=torch.float32)
        print("Creating test_hare . . . ")
        test_hare   = torch.tensor(np.array([hare[id_] for id_ in tqdm(test_ids)]), dtype=torch.float32)
        print("Creating verify_hare . . . ")
        verify_hare = torch.tensor(np.array([hare[id_] for id_ in tqdm(verify_ids)]), dtype=torch.float32)
        
        positions = geno.positions
        test_seqs = geno.tok_mat[test_all_pos,]
        test_phes = cancer[test_all_pos]
        train_seqs = geno.tok_mat[train_all_pos]
        train_phes = cancer[train_all_pos]
        verify_seqs = geno.tok_mat[verify_all_pos]
        verify_phes = cancer[verify_all_pos]
    
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        training_dataset = data.TensorDataset(
            # train_pheno_vec, 
            train_pcs,
            train_hare,
            train_age,
            train_gender,
            train_smoker,
            positions.repeat(len(train_seqs), 1),
            train_seqs, tensor(train_phes, dtype=torch.int64)
        )
        test_dataset = data.TensorDataset(
            # test_pheno_vec, 
            test_pcs,
            test_hare,
            test_age,
            test_gender,
            test_smoker,
            positions.repeat(len(test_seqs), 1),
            test_seqs, tensor(test_phes, dtype=torch.int64)
        )
        verify_dataset = data.TensorDataset(
            # verify_pheno_vec,
            verify_pcs,
            verify_hare,
            verify_age,
            verify_gender,
            verify_smoker,
            positions.repeat(len(verify_seqs), 1),
            verify_seqs, tensor(verify_phes, dtype=torch.int64)
        )
        return train_ids, training_dataset, test_ids, test_dataset, verify_ids, verify_dataset


def get_train_test_verify_ids(phenos, parameters):
    cache_dir = parameters['cache_dir']
    input_filtering = parameters['input_filtering']
    test_frac = parameters['test_frac']
    verify_frac = parameters['verify_frac']
    
    test_ids_file   = cache_dir + input_filtering + "-test_ids.csv"
    train_ids_file  = cache_dir + input_filtering + "-train_ids.csv"
    verify_ids_file = cache_dir + input_filtering + "-verify_ids.csv"
        
    if exists(test_ids_file) and exists(train_ids_file) and exists(verify_ids_file):
        if parameters['test_only']:
            test_ids = pandas.read_csv(test_ids_file)
        else:
            train_ids = pandas.read_csv(train_ids_file)
            test_ids = pandas.read_csv(test_ids_file)
            verify_ids = pandas.read_csv(verify_ids_file)
    else:
        if parameters['test_only']:        
            test_all_eids = phenos.eid
            test_ids   = pandas.DataFrame(test_all_eids)
            test_ids.to_csv(test_ids_file, header=["ID"], index=False)

        else:
            cancer = phenos["case"].values
            test_num_cancer = (int)(math.ceil(np.sum(cancer) * test_frac))
            verify_num_cancer = (int)(math.ceil(np.sum(cancer) * verify_frac))
            train_num_cancer = np.sum(cancer) - test_num_cancer - verify_num_cancer
            gp = phenos[phenos['case']==1]
            cp = phenos[phenos['case']==0]
        
            if input_filtering == "random_test_verify":
                new_cp = cp.iloc[np.random.choice(len(cp), len(gp), replace=False)]
        
            verify_num_cancer = (int)(math.ceil(len(gp) * verify_frac))
            train_num_cancer = len(gp) - test_num_cancer - verify_num_cancer
        
            gp_left = gp
            train_cancer_eid = np.random.choice(gp_left.eid, train_num_cancer, replace=False)
            gp_left = gp_left[~gp_left.eid.isin(train_cancer_eid)]
            test_cancer_eid = np.random.choice(gp_left.eid, test_num_cancer, replace=False)
            gp_left = gp_left[~gp_left.eid.isin(test_cancer_eid)]
            verify_cancer_eid = np.random.choice(gp_left.eid, verify_num_cancer, replace=False)
        
            cp_left = new_cp
            train_control_eid = np.random.choice(cp_left.eid, train_num_cancer, replace=False)
            cp_left = cp_left[~cp_left.eid.isin(train_control_eid)]
            test_control_eid = np.random.choice(cp_left.eid, test_num_cancer, replace=False)
            cp_left = cp_left[~cp_left.eid.isin(test_control_eid)]
            verify_control_eid = np.random.choice(cp_left.eid, verify_num_cancer, replace=False)
        
            train_cancer_eid.sort()
            test_cancer_eid.sort()
            verify_cancer_eid.sort()
            train_control_eid.sort()
            test_control_eid.sort()
            verify_control_eid.sort()
        
            if (not len(verify_control_eid) == len(verify_cancer_eid)):
                print("Warning! verification sets not balanced: {} (cancer) vs. {} (control)".format(len(verify_cancer_eid), len(verify_control_eid)))
        
        
            train_all_eids = np.concatenate([train_control_eid, train_cancer_eid])
            test_all_eids = np.concatenate([test_control_eid, test_cancer_eid])
            verify_all_eids = np.concatenate([verify_control_eid, verify_cancer_eid])
        
            train_ids  = pandas.DataFrame(train_all_eids)
            test_ids   = pandas.DataFrame(test_all_eids)
            verify_ids = pandas.DataFrame(verify_all_eids)
            train_ids.to_csv(train_ids_file, header=["ID"], index=False)
            test_ids.to_csv(test_ids_file, header=["ID"], index=False)
            verify_ids.to_csv(verify_ids_file, header=["ID"], index=False)

    if parameters['test_only']:
        test_ids = np.array(test_ids).reshape(-1)
        return test_ids
    else:
        train_ids = np.array(train_ids).reshape(-1)
        test_ids = np.array(test_ids).reshape(-1)
        verify_ids = np.array(verify_ids).reshape(-1)
        return train_ids, test_ids, verify_ids


def read_from_plink(parameters, phenos, remove_nan=False, subsample_control=False,
                test_frac=0.3, verify_frac=0.05,
                summarise_genos=False):
    print("using data from:", parameters['gwas_dir'])
    bed_file = parameters['gwas_dir']+parameters['plink_base']+".bed"
    bim_file = parameters['gwas_dir']+parameters['plink_base']+".bim"
    fam_file = parameters['gwas_dir']+parameters['plink_base']+".fam"
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    usable_ids = phenos["eid"].values.tolist()
    
    geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
    del geno_tmp

    print("getting train/test/verify split")
    if parameters['test_only']:
        test_ids = get_train_test_verify_ids(phenos, parameters)
    else:
        train_ids, test_ids, verify_ids = get_train_test_verify_ids(phenos, parameters)

    if (subsample_control):
        print("reducing to even control/non-control split")
        if parameters['test_only']:
            sample_ids = test_ids
        else:
            sample_ids = np.concatenate([train_ids, test_ids, verify_ids]).reshape(-1)
        phenos = phenos[phenos["eid"].isin(sample_ids)]
        geno = geno[geno["sample"].isin(sample_ids)]

    if summarise_genos:
        geno_mat = geno.values

        num_zeros = np.sum(geno_mat == 0)
        num_ones = np.sum(geno_mat == 1)
        num_twos = np.sum(geno_mat == 2)
        num_non_zeros = np.sum(geno_mat != 0)
        num_nan = np.sum(np.isnan(geno_mat))
        total_num = num_zeros + num_non_zeros
        values, counts = np.unique(geno_mat, return_counts=True)
        most_common = values[np.argmax(counts)]

        print(
            "geno mat contains {:.2f}% zeros, {:.2f}% ones, {:.2f}% twos {:.2f}% nans".format(
                100.0 * num_zeros / total_num,
                100 * (num_ones / total_num),
                100 * (num_twos / total_num),
                100.0 * num_nan / total_num,
            )
        )
        print("{:.2f}% has lung cancer".format(100 * np.sum(phenos.case) / len(phenos)))

    if remove_nan:
        geno_mat[np.isnan(geno_mat)] = most_common

    snv_toks = Tokenised_SNVs(geno)

    return snv_toks, phenos


def get_pretrain_dataset(phenos, params):
    print(phenos)
    pretrain_plink_base = params['pretrain_base']
    gwas_dir = params['gwas_dir']
    cache_dir = params['cache_dir']
    pt_pickle = cache_dir + "{}_pretrain.pickle".format(pretrain_plink_base)
    if (os.path.exists(pt_pickle)):
        with open(pt_pickle, "rb") as f:
            snv_toks = pickle.load(f)
    else:
        print("using data from:", gwas_dir)
        bed_file = gwas_dir + pretrain_plink_base + ".bed"
        bim_file = gwas_dir + pretrain_plink_base + ".bim"
        fam_file = gwas_dir + pretrain_plink_base + ".fam"
        print("bed_file:", bed_file)
        geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
        phenos = phenos.sort_values(by='eid')
        usable_ids = phenos["eid"].values.tolist()
        geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
        geno = geno.sortby("sample")
        print(len(geno.coords["sample"].values.tolist()))
        print(len(phenos["eid"].values.tolist()))
        print("Assserting that geno sample ids map one-to-one with pheno sample ids . . . ")
        assert all(geno["sample"].to_series().values==phenos["eid"].values)
        del geno_tmp
        snv_toks = Tokenised_SNVs(geno)
        with open(pt_pickle, "wb") as f:
            pickle.dump(snv_toks, f, pickle.HIGHEST_PROTOCOL)

    return snv_toks


class Tokenised_SNVs:
    def __init__(self, geno):
        tok_mat, tok_to_string, string_to_tok, num_toks = get_tok_mat(geno)
        self.snp_ids = geno.snp.values
        self.ids = geno.iid.values
        self.string_to_tok = string_to_tok
        self.tok_to_string = tok_to_string
        self.tok_mat = tok_mat
        self.num_toks = num_toks
        pos = {}
        for i,x in enumerate(sorted(geno.pos.values)):
            pos[(geno.snp.values[i], x)] = i
        pos_max = max(pos.values())
        exp = np.power(10,np.ceil(np.log10(pos_max)))
        tmp = np.array(geno.chrom.values).astype(int) * exp
        print(len(tmp))
        positions = torch.tensor(list(pos.values()), dtype=torch.long)
        new_pos = positions + tmp
        self.positions = new_pos
        
