import pickle
import torch
from torch import nan_to_num_, nn, tensor
from pytorch_attention import PositionalEncoding, LinformerAttention, D2LMultiHeadAttention, AddNorm, PositionWiseFFN

class TransformerBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, vocab_size, batch_size, device, use_linformer, linformer_k) -> None:
        super(TransformerBlock, self).__init__()
        if use_linformer:
            self.attention = LinformerAttention(embed_dim, seq_len, linformer_k, num_heads, dropout=0.05, use_sparsemax=False)
        else:
            self.attention = D2LMultiHeadAttention(embed_dim, num_heads, dropout=0.05, device=device)
        self.addnorm = AddNorm(embed_dim)
        self.positionwise_ffn = PositionWiseFFN(embed_dim, embed_dim, embed_dim)

    def forward(self, X):
        """X is (batch_size, seq_len, embed_dim)"""
        residual1 = X
        weights, at = self.attention(X, X, X, valid_lens=None)
        self.A = weights
        ra_out = self.addnorm(at, residual1)

        residual2 = ra_out
        ffn_out = self.positionwise_ffn(ra_out)
        return self.addnorm(ffn_out, residual2)


class FlattenedOutput(nn.Module):
    def __init__(self, embed_dim, seq_len, num_phenos, num_pcs) -> None:
        super().__init__()
        dense = nn.Linear(embed_dim*(seq_len+num_phenos)+num_pcs, 2)
        self.softmax = nn.Softmax(1)
        self.final_layer = nn.Sequential(dense, self.softmax)

    def forward(self, enc_out, pcs):
        cls_tok, phenos, seq_out = enc_out
        flat_out = torch.cat([cls_tok, phenos, seq_out, pcs], dim=1)
        flat_out = flat_out.view(flat_out.shape[0], -1)
        return self.final_layer(flat_out)


class TokenOutput(nn.Module):
    def __init__(self, embed_dim, pcs_dim) -> None:
        super().__init__()
        dense = nn.Linear(embed_dim+pcs_dim, 2)
        self.sigmoid = nn.Sigmoid()
        self.final_layer = nn.Sequential(dense, nn.GELU(), self.sigmoid)

    def forward(self, enc_out, pcs, USE_PCS = False):
        if USE_PCS:
            cls_tok = torch.cat([enc_out, pcs.to(enc_out.device)], dim=1).to(self.final_layer[0].weight.device)
        else:
            cls_tok = torch.cat([enc_out], dim=1).to(self.final_layer[0].weight.device)
            
        self.last_input = cls_tok
        return self.final_layer(cls_tok)


class Encoder(nn.Module):
    def __init__(self, params, cls_tok, vocab_size, max_seq_pos) -> None:
        super().__init__()
        seq_len = params['encoder_size']
        num_phenos = params['num_phenos']
        device = 0
        embed_dim = params['embed_dim']
        num_heads = params['num_heads']
        print("encoder vocab size: {}".format(vocab_size))
        self.cls_tok = cls_tok
        self.seq_len = seq_len
        self.num_phenos = num_phenos
        self.combined_seq_len = seq_len + num_phenos + 1
        self.device = device
        self.num_heads = num_heads
        self.pos_size = embed_dim - vocab_size
        self.use_phenos = params['use_phenos']
        if self.pos_size % 2 == 1:
            self.pos_size = self.pos_size - 1
        print("encoder 1-hot embedding size: {}".format(embed_dim))
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim).to('cuda:0')
        self.pos_encoding = PositionalEncoding(d_model=embed_dim, max_len=max_seq_pos + 1) ## with position info
        self.num_layers = params['num_layers']
        
        if self.num_layers>0:
            blocks = []
            for _ in range(self.num_layers):
                new_block = TransformerBlock(self.combined_seq_len,
                                             embed_dim,
                                             num_heads,
                                             vocab_size, params['batch_size'],
                                             device, params['use_linformer'],
                                             params['linformer_k'])
                blocks.append(new_block)
            self.encoder = nn.ModuleList(blocks)
        
    def forward(self, x):
        if self.num_layers>0:
            x = torch.cat([torch.tensor([[3]]*x.shape[0], dtype=torch.int8).to(x.device), x.long()],1)
        x = x.to(self.embedding.weight.device)
        ex = self.embedding(x)
        ex = self.pos_encoding(ex)
        
        if self.num_layers>0:
            for block in self.encoder:
                ex = block(ex)
                
        return ex


class TransformerModel(nn.Module):
    def __init__(self, encoder, params) -> None:
        super().__init__()
        self.encoder = encoder
        if torch.cuda.device_count()>1:
            embed_dim = self.encoder.module.embed_dim
        else:
            embed_dim = self.encoder.embed_dim
        
        output_type = params['output_type']
        num_phenos  = params['num_phenos']
        num_pcs     = params['num_pcs']
        seq_len     = params['encoder_size']
        self.num_layers  = params['num_layers']
        self.params = params
        
        if output_type == 'tok':
            self.output = TokenOutput(embed_dim, num_pcs)
        elif output_type == 'binary':
            self.output = FlattenedOutput(embed_dim, seq_len, num_phenos, num_pcs)
        else:
            raise ValueError("output_type must be 'binary', or 'tok'")

    def forward(self, x, pcs):
        enc_out = self.encoder(x)
        pcs = pcs.to(x.device)
        if self.num_layers>0:
            out = self.output(enc_out[:,0,:], pcs, USE_PCS = self.params['use_pcs'])
        else:
           out = self.output(enc_out, pcs, USE_PCS = self.params['use_pcs'])
        self.last_output = out
        return out