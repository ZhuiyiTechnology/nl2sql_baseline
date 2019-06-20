import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from sqlnet.model.modules.net_utils import run_lstm, col_name_encode

class SelNumPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, use_ca):
        super(SelNumPredictor, self).__init__()
        self.N_h = N_h
        self.use_ca = use_ca

        self.sel_num_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2), num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)

        self.sel_num_att = nn.Linear(N_h, 1)
        self.sel_num_col_att = nn.Linear(N_h, 1)
        self.sel_num_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(), nn.Linear(N_h,4))
        self.softmax = nn.Softmax(dim=-1)
        self.sel_num_col2hid1 = nn.Linear(N_h, 2 * N_h)
        self.sel_num_col2hid2 = nn.Linear(N_h, 2 * N_h)


        if self.use_ca:
            print ("Using column attention on select number predicting")

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num):
        B = len(x_len)
        max_x_len = max(x_len)

        # Predict the number of select part
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict select number
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                                             col_len, self.sel_num_lstm)
        num_col_att_val = self.sel_num_col_att(e_num_col).squeeze()
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -1000000
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        sel_num_h1 = self.sel_num_col2hid1(K_num_col).view((B, 4, self.N_h//2)).transpose(0,1).contiguous()
        sel_num_h2 = self.sel_num_col2hid2(K_num_col).view((B, 4, self.N_h//2)).transpose(0,1).contiguous()

        h_num_enc, _ = run_lstm(self.sel_num_lstm, x_emb_var, x_len,hidden=(sel_num_h1, sel_num_h2))

        num_att_val = self.sel_num_att(h_num_enc).squeeze()
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -1000000
        num_att = self.softmax(num_att_val)

        K_sel_num = (h_num_enc * num_att.unsqueeze(2).expand_as(h_num_enc)).sum(1)
        sel_num_score = self.sel_num_out(K_sel_num)
        return sel_num_score







