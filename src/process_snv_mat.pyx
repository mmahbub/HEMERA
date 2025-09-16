import torch
import numpy as np
import pickle
import math
from cython.parallel import prange
from tqdm import tqdm
cimport cython


def enc(string_to_tok, tok_to_string, pos, p: int, a0, a1):
    a0_toks = np.zeros(p, dtype=np.int32)
    a1_toks = np.zeros(p, dtype=np.int32)
    a01_toks = np.zeros(p, dtype=np.int32)
    for i,string in enumerate(a0):
        if (len(string) > 1):
            string = string[0] + 'I'
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a0_toks[i] = tok
    for i in range(len(a1)):
        if len(a1[i]) == len(a0[i]):
            if (len(a1[i]) > 1):
                string = a1[i][0] + 'I'
            else:
                string = a1[i]
        elif len(a1[i]) > len(a0[i]):
            string = 'ins'
        elif len(a1[i]) < len(a0[i]):
            string = 'del'
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a1_toks[i] = tok
    for i,(a,b) in enumerate(zip(a0,a1)):
        a = str(a)
        b = str(b)
        a_len = len(a)
        b_len = len(b)
        if (a_len > 1):
            a = a[0] + 'I'
        if (b_len > 1):
            b = b[0] + 'I'
        if b_len == a_len:
            string = a + ',' + b
        elif b_len > a_len:
            string = a + ',' + 'ins'
        elif b_len < a_len:
            string = a + ',' + 'del'
        if string in string_to_tok:
            tok = string_to_tok[string]
        else:
            tok = pos
            string_to_tok[string] = pos
            tok_to_string[pos] = string
            pos = pos + 1
        a01_toks[i] = tok
    return a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok


@cython.boundscheck(False) 
@cython.wraparound(False) 
def get_tok_mat(geno):
    a0 = geno.a0.values
    a1 = geno.a1.values
    encoded_tok_list = []
    
    # string_to_tok = {}
    # tok_to_string = {}

    # loading saved tokens
    string_to_tok = pickle.load(open("../cache/string_to_tok.pkl","rb")) # need to save it the first time you run the code
    tok_to_string = pickle.load(open("../cache/tok_to_string.pkl","rb")) # need to save it the first time you run the code
 
    (n_tmp, p_tmp) = geno.shape
    cdef int n = n_tmp
    cdef int p = p_tmp
    cdef int pos = 0

    # for special_tok in ['nan', 'ins', 'del', 'cls', 'mask']:
    #     string_to_tok[special_tok] = pos
    #     tok_to_string[pos] = special_tok
        # pos = pos + 1

    cdef int nan_tok = string_to_tok['nan']

    cdef unsigned char tok

    pos = len(string_to_tok)
    print("identifying tokens")

    diff_lens = None

    a0_toks, a1_toks, a01_toks, tok_to_string, string_to_tok = enc(string_to_tok, tok_to_string, pos, p, a0, a1)

    tok_mat = np.zeros((n, p), dtype=np.uint8)
    alleles_differ_mat = np.zeros((n,p), dtype=np.bool_) # N.B. stored as a byte.
    is_nonref_mat = np.zeros((n,p), dtype=np.bool_) # N.B. stored as a byte.

    # memory views for numpy arrays
    cdef int [:] a0_toks_view = a0_toks
    cdef int [:] a1_toks_view = a1_toks
    cdef int [:] a01_toks_view = a01_toks
    cdef unsigned char [:,:] alleles_differ_mat_view = alleles_differ_mat
    cdef unsigned char [:,:] is_nonref_mat_view = is_nonref_mat
    cdef unsigned char [:,:] tok_mat_view = tok_mat
    # cdef int [:,:] geno_mat_view = geno_mat

    print("building token matrix")

    cdef int batch_size = 32
    cdef Py_ssize_t ri, ind
    cdef int val
    cdef int [:,:] geno_mat_view
    cdef int actual_row = 0
    cdef int batch
    cdef int actual_batch_len
    for batch in tqdm(range(int(np.ceil(float(n)/batch_size)))):
        geno_mat = np.array(geno[batch*batch_size:(batch+1)*batch_size].values, dtype=np.int32)
        geno_mat_view = geno_mat
        actual_batch_len = geno_mat.shape[0]
        with nogil:
            for ri in prange(actual_batch_len):
                actual_row = batch * batch_size + ri
                if actual_row < n:
                    for ind in range(p):
                        val = geno_mat_view[ri, ind]
                        if val == 0:
                            tok = a0_toks_view[ind]
                        elif val == 2:
                            tok = a1_toks_view[ind]
                        elif val == 1:
                            tok = a01_toks_view[ind]
                        else:
                            tok = nan_tok
                        tok_mat_view[actual_row, ind] = tok

    tok_mat = torch.from_numpy(tok_mat)
    return tok_mat, tok_to_string, string_to_tok, len(string_to_tok)
