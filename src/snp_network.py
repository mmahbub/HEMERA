import torch
from torch import nn
from torch.utils import data
import math
import pickle
from os.path import exists
import numpy as np
from snp_input import get_data, get_pretrain_dataset, Tokenised_SNVs
from self_attention_net import Encoder, TransformerModel
from tqdm import tqdm
from collections import Counter, OrderedDict


def get_pretrained_encoder(state_file: str, params, snv_toks):
    print("attempting to load encoder from file {}".format(state_file))
    encoder = get_encoder(params, snv_toks)
    # encoder = nn.DataParallel(encoder, [0])
    encoder.load_state_dict(torch.load(state_file))
    return encoder


def get_encoder_from_params(p: dict, pretrain_snv_toks):
    return get_encoder(p['encoder_size'], p['num_phenos'], pretrain_snv_toks.positions.max(), pretrain_snv_toks.num_toks, p['batch_size'], 0, pretrain_snv_toks.string_to_tok['cls'], p)


def get_encoder(params, pretrain_snv_toks: Tokenised_SNVs):
    cls_tok = pretrain_snv_toks.string_to_tok['cls']
    vocab_size = pretrain_snv_toks.num_toks
    max_seq_pos = pretrain_snv_toks.positions.max()
    encoder = Encoder(
        params,
        cls_tok,
        vocab_size,
        max_seq_pos,
    )
    print(encoder)
    return encoder


def transformer_from_encoder(encoder, params):
    net = TransformerModel(encoder, params)
    print(net)
    return net


def get_transformer(params, snv_toks):
    net = TransformerModel(
        get_encoder(params, snv_toks), params)
    print(net)
    return net


def mask_sequence(seqs, frac, tokenised_snvs: Tokenised_SNVs):
    rng = np.random.default_rng()
    mask_tok = tokenised_snvs.string_to_tok["mask"]
    # ratios are from BERT
    frac_masked = 0.8
    frac_random = 0.1
    frac_pos_masked = 0.1
    mask_positions = []
    random_positions = []
    mask_pos_indices = []

    for seq in range(seqs.shape[0]):
        pos_inds = rng.choice(seqs.shape[1], size=(int)(math.ceil(frac_pos_masked*seqs.shape[1])), replace=False)
        seq_change_positions = rng.choice(seqs.shape[1], size=(int)(math.ceil(frac*seqs.shape[1])), replace=False)
        seq_mask_positions = [(seq,i) for i in rng.choice(seq_change_positions, (int)(math.ceil(frac_masked*len(seq_change_positions))), replace=False)]
        seq_random_positions = [(seq,i) for i in rng.choice(seq_change_positions, (int)(math.ceil(frac_random*len(seq_change_positions))), replace=False)]
        mask_positions.extend(seq_mask_positions)
        random_positions.extend(seq_random_positions)
        seq_pos_mask_inds = [(seq,i) for i in pos_inds]
        mask_pos_indices.extend(seq_pos_mask_inds)
    # replace mask tokens
    mask_positions = np.transpose(np.array(mask_positions))
    random_positions = np.transpose(np.array(random_positions))
    mask_pos_indices = np.transpose(np.array(mask_pos_indices))
    seqs[tuple(mask_positions)] = mask_tok
    # replace random tokens
    new_random_toks = torch.tensor(rng.choice([i for i in tokenised_snvs.tok_to_string.keys()], random_positions.shape[1], replace=True), dtype=torch.uint8)
    seqs[tuple(random_positions)] = new_random_toks
    return seqs

def pretrain_encoder(
    encoder, pretrain_snv_toks, batch_size, num_epochs,
        device, learning_rate, pretrain_base, params, pretrain_log_file):

    print("beginning pre-training encoder")
    positions = pretrain_snv_toks.positions
    pos_seq_dataset = data.TensorDataset(positions.repeat(pretrain_snv_toks.tok_mat.shape[0], 1), pretrain_snv_toks.tok_mat)
    training_iter = data.DataLoader(pos_seq_dataset, batch_size, shuffle=True)

    for param in encoder.parameters():
        if param.device.type == 'cpu':
            print(param.device)
            print(param.shape)
            break
    trainer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, amsgrad=True)
    loss = nn.CrossEntropyLoss()

    class_size = 34
    rng = np.random.default_rng()

    #start training 
    for e in range(num_epochs):
        sum_loss = 0.0
        num_steps = 0
        for pos, seqs in tqdm(training_iter, ncols=0):
            if torch.cuda.device_count()>1:
                input_size = encoder.module.seq_len
            else:
                input_size = encoder.seq_len
            chosen_positions = rng.choice(seqs.shape[1], input_size, replace=False)
            chosen_positions.sort()
            seqs = seqs[:,chosen_positions]
            masked_seqs = mask_sequence(seqs, 0.40, pretrain_snv_toks)
            masked_seqs = masked_seqs.to(device)
            pred_seqs = encoder(masked_seqs)
            pred_class_probs = torch.softmax(pred_seqs[:,:,0:class_size], dim=2)
            pred_class_probs = torch.swapdims(pred_class_probs, 1, 2)
            true_classes = seqs.long().to(device)
            l = loss(pred_class_probs[:,:,1:], true_classes)

            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            sum_loss += l.mean()
            num_steps = num_steps + 1

        if torch.cuda.device_count()>1:
            encoder_file = params['saved_nets_dir']+"pretrain_{}_epoch-{}_encoder_encsize-{}.pickle".format(
                pretrain_base, e, encoder.module.seq_len
            )
        else:
            encoder_file = params['saved_nets_dir']+"pretrain_{}_epoch-{}_encoder_encsize-{}.pickle".format(
                pretrain_base, e, encoder.seq_len
            )
        if torch.cuda.device_count()>1:
            torch.save(encoder.module.state_dict(), encoder_file)
        else:
            torch.save(encoder.state_dict(), encoder_file)
        s = "epoch {}, loss {}".format(e, sum_loss/num_steps)
        print(s)
        pretrain_log_file.write(s)


def subsample_forward(net, seqs, pcs, device):
    rng = np.random.default_rng()
    if torch.cuda.device_count()>1:
        input_size = net.module.encoder.module.seq_len
    else:
        input_size = net.encoder.seq_len
    chosen_positions = rng.choice(seqs.shape[1], input_size, replace=False)
    chosen_positions.sort()
    seqs = seqs[:, chosen_positions]
    seqs = seqs.to(device)
    return net(seqs, pcs)


def mask_forward(net, seqs: torch.Tensor, pcs, device, snv_toks: Tokenised_SNVs):
    if torch.cuda.device_count()>1:
        encoder_size = net.module.encoder.module.seq_len
    else:
        encoder_size = net.encoder.seq_len
    mask_tok = snv_toks.string_to_tok['mask']
    orig_seq_len = seqs.shape[1]
    expanded_seqs = seqs.expand(seqs.shape[0], encoder_size)
    expanded_seqs[:, orig_seq_len:] = mask_tok
    return net(expanded_seqs, pcs)

def mask_all_genos_forward(net, seqs, pcs, snv_toks):
    mask_seqs = torch.tensor(snv_toks.string_to_tok['mask']).expand(seqs.shape[0], seqs.shape[1])
    return net(mask_seqs, pcs)
    
def process_input(subsample, net, seqs, pcs, device, snv_toks, params):
    if torch.cuda.device_count()>1:
        encoder_size = net.module.encoder.module.seq_len
    else:
        encoder_size = net.encoder.seq_len
        
    if params['mask_genotypes']:
        return mask_all_genos_forward(net, seqs, pcs, snv_toks)
    else:
        if seqs.shape[1] < encoder_size:
            return mask_forward(net, seqs, pcs, device, snv_toks)
        if subsample:
            return subsample_forward(net, seqs, pcs, device)
        else:
            return net(seqs, pcs)


def train_net(
    net, training_dataset, test_dataset, batch_size, num_epochs,
        device, learning_rate, prev_num_epochs, test_split, train_log_file, plink_base, snv_toks, params, subsample_input=False
):
    training_iter = data.DataLoader(training_dataset, batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, int(batch_size), shuffle=True)
    trainer = torch.optim.AdamW(net.parameters(), lr=learning_rate, amsgrad=True)

    class_label = [str(int(y)) for _, _, _, _, _, _, _, y in tqdm(training_dataset)]
    weights = Counter(class_label)
    weights = [weights['1']/len(class_label),weights['0']/len(class_label)] #[0.498445, 0.501555]
    class_weights = torch.tensor(weights).to('cuda:0')
    print(f"class_weights: {class_weights}")
    
    loss = nn.CrossEntropyLoss(weight=class_weights)

    patience = 5
    min_delta = 1e-4
    best_validation_loss = float('inf')
    epochs_without_improvement = 0
    
    print("starting training")
    for e in range(num_epochs):
        sum_loss = 0.0
        for pcs, _, _, _, _, pos, X, y in tqdm(training_iter, ncols=0):
            Yh = process_input(subsample_input, net, X, pcs, device, snv_toks, params)
            l = loss(Yh, y.to(Yh.device))
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            sum_loss += l.mean()
        if e % 2 == 0:
            test_loss = 0.0
            with torch.no_grad():
                for pcs, _, _, _, _, pos, X, y in tqdm(test_iter, ncols=0):
                    Yh = process_input(subsample_input, net, X, pcs, device, snv_toks, params)
                    l = loss(Yh, y.to(Yh.device))
                    test_loss += l.mean()
            
            avg_validation_loss = test_loss / len(test_iter)
            
            tmpstr = "epoch {}, mean loss {:.5}, {:.5} (test)".format(
                    e, sum_loss / len(training_iter), test_loss / len(test_iter))
            print(tmpstr)
            train_log_file.write(tmpstr)

            # Early stopping check
            if best_validation_loss - avg_validation_loss >= min_delta:
                print(f"Best validation loss so far: {best_validation_loss}")
                print(f"New validation loss: {avg_validation_loss}")
                best_validation_loss = avg_validation_loss
                if torch.cuda.device_count()>1:
                    best_model_weights = net.module.state_dict()
                else:
                    best_model_weights = net.state_dict()
                print("Model improved. Best model weights saved.")
            else:
                epochs_without_improvement+=1
                print(f"No improvement for {epochs_without_improvement}/{patience} epochs")
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {e} due to no improvement")
                    break
            
            net_file = params['saved_nets_dir']+"{}_epoch-{}".format(get_net_savename(params), e)
            if torch.cuda.device_count()>1:
                torch.save(net.module.state_dict(), net_file)
            else:
                torch.save(net.state_dict(), net_file)

    # load the best model
    if torch.cuda.device_count()>1:
        new_state_dict = OrderedDict()
        for k,v in best_model_weights.items():
            new_state_dict[f"module.{k}"] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(best_model_weights)

    # start testing
    test_total_correct = 0.0
    test_total_incorrect = 0.0
    sum_loss = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for pcs, _, _, _, _, pos, tX, tY in test_iter:
            tYh = process_input(subsample_input, net, tX, pcs, device, snv_toks, params)
            binary_tYh = tYh[:, 1] > 0.5
            binary_tY = tY > 0.5
            binary_tY = binary_tY.to(device)
            correct = binary_tYh == binary_tY
            num_correct = torch.sum(correct)
            test_total_correct += num_correct
            test_total_incorrect += len(tX) - num_correct
            l = loss(tYh, tY.to(tYh.device))
            test_loss += l.mean()
    train_total_correct = 0.0
    train_total_incorrect = 0.0
    with torch.no_grad():
        for pcs, _, _, _, _, pos, tX, tY in training_iter:
            tYh = process_input(subsample_input, net, tX, pcs, device, snv_toks, params)
            binary_tYh = tYh[:, 1] > 0.5
            binary_tY = tY > 0.5
            binary_tY = binary_tY.to(device)
            correct = binary_tYh == binary_tY
            num_correct = torch.sum(correct)
            train_total_correct += num_correct
            train_total_incorrect += len(tX) - num_correct
            l = loss(tYh, tY.to(tYh.device))
            sum_loss += l.mean()
    tmpstr = "epoch {}, mean loss {:.3}, {:.3} (test)".format(
            e, sum_loss / len(training_iter), test_loss / len(test_iter))
    print(tmpstr)
    train_log_file.write(tmpstr)
    tmpstr = "final fraction correct: {:.3f} (train), {:.3f} (test)".format(
            train_total_correct /
            (train_total_correct + train_total_incorrect),
            test_total_correct / (test_total_correct + test_total_incorrect),
        )
    print(tmpstr)
    train_log_file.write(tmpstr)

    return net


def check_pos_neg_frac(dataset: data.TensorDataset):
    pos, neg, unknown = 0, 0, 0
    for (_, _, _, _, _, _, _, y) in dataset:
        if y == 1:
            pos = pos + 1
        elif y == 0:
            neg = neg + 1
        else:
            unknown = unknown + 1
    return pos, neg, unknown


default_parameters = {
    'mask_genotypes': False,

}


def get_net_savename(parameters: dict):
    par_str = "_".join("{}{}".format(k,v) for k, v in parameters.items() if k in ["num_heads", "num_layers", "batch_size", "num_epochs", "pretrain_epochs"])    
    par_str += '.net'
    return par_str.replace('/', '-')


def train_everything(phenos,params=default_parameters):
    pretrain_base = params['pretrain_base']
    plink_base = params['plink_base']
    continue_training = params['continue_training']
    train_new_encoder = params['train_new_encoder']
    test_frac = params['test_frac']
    verify_frac = params['verify_frac']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    lr = params['lr']
    output = params['output_type']
    encoder_size = params['encoder_size']
    pt_epochs = params['pretrain_epochs']
    pt_lr = params['pt_lr']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ids, train, test_ids, test, verify_ids, verify, geno, pheno = get_data(phenos,params)
    snv_toks = geno

    net_dir = params['saved_nets_dir']

    new_epoch = num_epochs
    new_net_name = net_dir + "{}_batch-{}_epochs-{}_p-{}_n-{}_epoch-{}_test_split-{}_output-{}_net.pickle".format(
        plink_base, batch_size, num_epochs,
        geno.tok_mat.shape[1], geno.tok_mat.shape[0], new_epoch,
        test_frac, output
    )

    # Pre-training
    pretrain_snv_toks = get_pretrain_dataset(train_ids, params)

    print("geno num toks: {}".format(geno.num_toks))
    print("pretrain_snv_toks num toks: {}".format(pretrain_snv_toks.num_toks))
    geno_toks = set([*geno.tok_to_string.keys()])
    pt_toks = set([*pretrain_snv_toks.tok_to_string.keys()])
    ft_new_toks = geno_toks - geno_toks.intersection(pt_toks)
    if (len(ft_new_toks) > 0):
        print([geno.tok_to_string[i] for i in ft_new_toks])
        raise Exception("Warning, fine tuning set contains new tokens!")

    if (encoder_size == -1):
        encoder_size = geno.tok_mat.shape[1]
    print("creating encoder w/ input size: {}".format(encoder_size))
    encoder_params = {k:params[k] for k in('pretrain_base', 'test_frac', 'verify_frac', 'batch_size', 'pt_lr', 'encoder_size', 'num_phenos', 'pretrain_epochs', 'input_filtering')}
    encoder_file = net_dir + get_net_savename(encoder_params) + ".encoder"

    if (train_new_encoder or not exists(encoder_file)):
        pt_batch_size = batch_size
        encoder = get_encoder(params, pretrain_snv_toks)

        # add data parallelism
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        if torch.cuda.device_count()>1:
            encoder = torch.nn.DataParallel(encoder)
        pt_net_name = "bs-{}_epochs-{}_lr-{}_pretrained.net".format(pt_batch_size, pt_epochs, pt_lr)
        pt_log_file = open(pt_net_name + ".log", "w")
        print("pre-training encoder with sequences of length {}".format(pretrain_snv_toks.tok_mat.shape[1]))
        pretrain_encoder(encoder, pretrain_snv_toks, pt_batch_size, pt_epochs, device, pt_lr, pretrain_base, params, pt_log_file)
        pt_log_file.close()
        if torch.cuda.device_count()>1:
            torch.save(encoder.module.state_dict(), encoder_file)
        else:
            torch.save(encoder.state_dict(), encoder_file)
    else:
        encoder = get_pretrained_encoder(encoder_file, params, pretrain_snv_toks)
        encoder = encoder.to(device)
        if torch.cuda.device_count()>1:
            encoder = torch.nn.DataParallel(encoder)
        
    # Fine-tuning
    net = transformer_from_encoder(encoder, params)

    # net = nn.DataParallel(net, use_device_ids)
    if (continue_training):
        prev_params = params['prev_params']
        prev_net_name = net_dir + get_net_savename(prev_params)
        net.load_state_dict(torch.load(prev_net_name))
        new_net_name = net_dir + get_net_savename(params)
    else:
        prev_epoch = 0

    # net = net.to(use_device_ids[0])

    # add data parallelism
    net = net.to('cuda:0')
    if torch.cuda.device_count()>1:
        net = torch.nn.DataParallel(net)
    
    train_log_file = open(new_net_name + "_log.txt", "w")

    print("train dataset:  ", check_pos_neg_frac(train))
    print("test dataset:   ", check_pos_neg_frac(test))
    print("verify dataset: ", check_pos_neg_frac(verify))
    net = train_net(net, train, verify, batch_size, num_epochs, device, lr, prev_epoch, test_frac, train_log_file, plink_base, snv_toks, params, subsample_input=False)

    train_log_file.close()

    if torch.cuda.device_count()>1:
        torch.save(net.module.state_dict(), new_net_name)
    else:
        torch.save(net.state_dict(), new_net_name)
    return net

if __name__ == "__main__":
    train_everything()
