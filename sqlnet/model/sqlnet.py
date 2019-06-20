import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sqlnet.model.modules.word_embedding import WordEmbedding
from sqlnet.model.modules.aggregator_predict import AggPredictor
from sqlnet.model.modules.selection_predict import SelPredictor
from sqlnet.model.modules.sqlnet_condition_predict import SQLNetCondPredictor
from sqlnet.model.modules.select_number import SelNumPredictor
from sqlnet.model.modules.where_relation import WhereRelationPredictor


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
            gpu=False, use_ca=True, trainable_emb=False):
        super(SQLNet, self).__init__()
        self.use_ca = use_ca
        self.trainable_emb = trainable_emb

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'OR', '==', '>', '<', '!=', '<BEG>']
        self.COND_OPS = ['>', '<', '==', '!=']

        # Word embedding
        self.embed_layer = WordEmbedding(word_emb, N_word, gpu, self.SQL_TOK, our_model=True, trainable=trainable_emb)

        # Predict the number of selected columns
        self.sel_num = SelNumPredictor(N_word, N_h, N_depth, use_ca=use_ca)

        #Predict which columns are selected
        self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num, use_ca=use_ca)

        #Predict aggregation functions of corresponding selected columns
        self.agg_pred = AggPredictor(N_word, N_h, N_depth, use_ca=use_ca)

        #Predict number of conditions, condition columns, condition operations and condition values
        self.cond_pred = SQLNetCondPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, use_ca, gpu)

        # Predict condition relationship, like 'and', 'or'
        self.where_rela_pred = WhereRelationPredictor(N_word, N_h, N_depth, use_ca=use_ca)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()

    def generate_gt_where_seq_test(self, q, gt_cond_seq):
        ret_seq = []
        for cur_q, ans in zip(q, gt_cond_seq):
            temp_q = u"".join(cur_q)
            cur_q = [u'<BEG>'] + cur_q + [u'<END>']
            record = []
            record_cond = []
            for cond in ans:
                if cond[2] not in temp_q:
                    record.append((False, cond[2]))
                else:
                    record.append((True, cond[2]))
            for idx, item in enumerate(record):
                temp_ret_seq = []
                if item[0]:
                    temp_ret_seq.append(0)
                    temp_ret_seq.extend(list(range(temp_q.index(item[1])+1,temp_q.index(item[1])+len(item[1])+1)))
                    temp_ret_seq.append(len(cur_q)-1)
                else:
                    temp_ret_seq.append([0,len(cur_q)-1])
                record_cond.append(temp_ret_seq)
            ret_seq.append(record_cond)
        return ret_seq

    def forward(self, q, col, col_num, gt_where = None, gt_cond=None, reinforce=False, gt_sel=None, gt_sel_num=None):
        B = len(q)

        sel_num_score = None
        agg_score = None
        sel_score = None
        cond_score = None
        #Predict aggregator
        if self.trainable_emb:
            x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col)
            col_inp_var, col_name_len, col_len = self.agg_embed_layer.gen_col_batch(col)
            max_x_len = max(x_len)
            agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var,
                    col_name_len, col_len, col_num, gt_sel=gt_sel)

            x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col)
            col_inp_var, col_name_len, col_len = self.sel_embed_layer.gen_col_batch(col)
            max_x_len = max(x_len)
            sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                    col_name_len, col_len, col_num)

            x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
            col_inp_var, col_name_len, col_len = self.cond_embed_layer.gen_col_batch(col)
            max_x_len = max(x_len)
            cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_where, gt_cond, reinforce=reinforce)
            where_rela_score = None
        else:
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
            col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)
            sel_num_score = self.sel_num(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)
            # x_emb_var: embedding of each question
            # x_len: length of each question
            # col_inp_var: embedding of each header
            # col_name_len: length of each header
            # col_len: number of headers in each table, array type
            # col_num: number of headers in each table, list type
            if gt_sel_num:
                pr_sel_num = gt_sel_num
            else:
                pr_sel_num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1)
            sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

            if gt_sel:
                pr_sel = gt_sel
            else:
                num = np.argmax(sel_num_score.data.cpu().numpy(), axis=1)
                sel = sel_score.data.cpu().numpy()
                pr_sel = [list(np.argsort(-sel[b])[:num[b]]) for b in range(len(num))]
            agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_sel=pr_sel, gt_sel_num=pr_sel_num)

            where_rela_score = self.where_rela_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

            cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_where, gt_cond, reinforce=reinforce)

        return (sel_num_score, sel_score, agg_score, cond_score, where_rela_score)

    def loss(self, score, truth_num, gt_where):
        sel_num_score, sel_score, agg_score, cond_score, where_rela_score = score

        B = len(truth_num)
        loss = 0

        # Evaluate select number
        # sel_num_truth = map(lambda x:x[0], truth_num)
        sel_num_truth = [x[0] for x in truth_num]
        sel_num_truth = torch.from_numpy(np.array(sel_num_truth))
        if self.gpu:
            sel_num_truth = Variable(sel_num_truth.cuda())
        else:
            sel_num_truth = Variable(sel_num_truth)
        loss += self.CE(sel_num_score, sel_num_truth)

        # Evaluate select column
        T = len(sel_score[0])
        truth_prob = np.zeros((B,T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][list(truth_num[b][1])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            sel_col_truth_var = Variable(data.cuda())
        else:
            sel_col_truth_var = Variable(data)
        sigm = nn.Sigmoid()
        sel_col_prob = sigm(sel_score)
        bce_loss = -torch.mean(
            3*(sel_col_truth_var * torch.log(sel_col_prob+1e-10)) +
            (1-sel_col_truth_var) * torch.log(1-sel_col_prob+1e-10)
        )
        loss += bce_loss

        # Evaluate select aggregation
        for b in range(len(truth_num)):
            data = torch.from_numpy(np.array(truth_num[b][2]))
            if self.gpu:
                sel_agg_truth_var = Variable(data.cuda())
            else:
                sel_agg_truth_var = Variable(data)
            sel_agg_pred = agg_score[b, :len(truth_num[b][1])]
            loss += (self.CE(sel_agg_pred, sel_agg_truth_var)) / len(truth_num)

        cond_num_score, cond_col_score, cond_op_score, cond_str_score = cond_score

        # Evaluate the number of conditions
        # cond_num_truth = map(lambda x:x[3], truth_num)
        cond_num_truth = [x[3] for x in truth_num]
        data = torch.from_numpy(np.array(cond_num_truth))
        if self.gpu:
            try:
                cond_num_truth_var = Variable(data.cuda())
            except:
                print ("cond_num_truth_var error")
                print (data)
                exit(0)
        else:
            cond_num_truth_var = Variable(data)
        loss += self.CE(cond_num_score, cond_num_truth_var)

        # Evaluate the columns of conditions
        T = len(cond_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][4]) > 0:
                truth_prob[b][list(truth_num[b][4])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            cond_col_truth_var = Variable(data.cuda())
        else:
            cond_col_truth_var = Variable(data)

        sigm = nn.Sigmoid()
        cond_col_prob = sigm(cond_col_score)
        bce_loss = -torch.mean(
            3*(cond_col_truth_var * torch.log(cond_col_prob+1e-10)) +
            (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
        loss += bce_loss

        # Evaluate the operator of conditions
        for b in range(len(truth_num)):
            if len(truth_num[b][5]) == 0:
                continue
            data = torch.from_numpy(np.array(truth_num[b][5]))
            if self.gpu:
                cond_op_truth_var = Variable(data.cuda())
            else:
                cond_op_truth_var = Variable(data)
            cond_op_pred = cond_op_score[b, :len(truth_num[b][5])]
            try:
                loss += (self.CE(cond_op_pred, cond_op_truth_var) / len(truth_num))
            except:
                print (cond_op_pred)
                print (cond_op_truth_var)
                exit(0)

        #Evaluate the strings of conditions
        for b in range(len(gt_where)):
            for idx in range(len(gt_where[b])):
                cond_str_truth = gt_where[b][idx]
                if len(cond_str_truth) == 1:
                    continue
                data = torch.from_numpy(np.array(cond_str_truth[1:]))
                if self.gpu:
                    cond_str_truth_var = Variable(data.cuda())
                else:
                    cond_str_truth_var = Variable(data)
                str_end = len(cond_str_truth)-1
                cond_str_pred = cond_str_score[b, idx, :str_end]
                loss += (self.CE(cond_str_pred, cond_str_truth_var) \
                        / (len(gt_where) * len(gt_where[b])))

        # Evaluate condition relationship, and / or
        # where_rela_truth = map(lambda x:x[6], truth_num)
        where_rela_truth = [x[6] for x in truth_num]
        data = torch.from_numpy(np.array(where_rela_truth))
        if self.gpu:
            try:
                where_rela_truth = Variable(data.cuda())
            except:
                print ("where_rela_truth error")
                print (data)
                exit(0)
        else:
            where_rela_truth = Variable(data)
        loss += self.CE(where_rela_score, where_rela_truth)
        return loss

    def check_acc(self, vis_info, pred_queries, gt_queries):
        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                    self.COND_OPS[cond[1]] + ' ' + unicode(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        tot_err = sel_num_err = agg_err = sel_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            sel_pred, agg_pred, where_rela_pred = pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op']
            sel_gt, agg_gt, where_rela_gt = gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op']

            if where_rela_gt != where_rela_pred:
                good = False
                cond_rela_err += 1

            if len(sel_pred) != len(sel_gt):
                good = False
                sel_num_err += 1

            pred_sel_dict = {k:v for k,v in zip(list(sel_pred), list(agg_pred))}
            gt_sel_dict = {k:v for k,v in zip(sel_gt, agg_gt)}
            if set(sel_pred) != set(sel_gt):
                good = False
                sel_err += 1
            agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
            agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
            if agg_pred != agg_gt:
                good = False
                agg_err += 1

            cond_pred = pred_qry['conds']
            cond_gt = gt_qry['conds']
            if len(cond_pred) != len(cond_gt):
                good = False
                cond_num_err += 1
            else:
                cond_op_pred, cond_op_gt = {}, {}
                cond_val_pred, cond_val_gt = {}, {}
                for p, g in zip(cond_pred, cond_gt):
                    cond_op_pred[p[0]] = p[1]
                    cond_val_pred[p[0]] = p[2]
                    cond_op_gt[g[0]] = g[1]
                    cond_val_gt[g[0]] = g[2]

                if set(cond_op_pred.keys()) != set(cond_op_gt.keys()):
                    cond_col_err += 1
                    good=False

                where_op_pred = [cond_op_pred[x] for x in sorted(cond_op_pred.keys())]
                where_op_gt = [cond_op_gt[x] for x in sorted(cond_op_gt.keys())]
                if where_op_pred != where_op_gt:
                    cond_op_err += 1
                    good=False

                where_val_pred = [cond_val_pred[x] for x in sorted(cond_val_pred.keys())]
                where_val_gt = [cond_val_gt[x] for x in sorted(cond_val_gt.keys())]
                if where_val_pred != where_val_gt:
                    cond_val_err += 1
                    good=False

            if not good:
                tot_err += 1

        return np.array((sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err , cond_rela_err)), tot_err


    def gen_query(self, score, q, col, raw_q, reinforce=False, verbose=False):
        """
        :param score:
        :param q: token-questions
        :param col: token-headers
        :param raw_q: original question sequence
        :return:
        """
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str# .lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(',
                    '-RRB-':')',
                    '-LSB-':'[',
                    '-RSB-':']',
                    '``':'"',
                    '\'\'':'"',
                    '--':u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear
                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                # elif tok[0] not in alphabet:
                #     pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        sel_num_score, sel_score, agg_score, cond_score, where_rela_score = score
        # [64,4,6], [64,14], ..., [64,4]
        sel_num_score = sel_num_score.data.cpu().numpy()
        sel_score = sel_score.data.cpu().numpy()
        agg_score = agg_score.data.cpu().numpy()
        where_rela_score = where_rela_score.data.cpu().numpy()
        ret_queries = []
        B = len(agg_score)
        cond_num_score,cond_col_score,cond_op_score,cond_str_score =\
            [x.data.cpu().numpy() for x in cond_score]
        for b in range(B):
            cur_query = {}
            cur_query['sel'] = []
            cur_query['agg'] = []
            sel_num = np.argmax(sel_num_score[b])
            max_col_idxes = np.argsort(-sel_score[b])[:sel_num]
            # find the most-probable columns' indexes
            max_agg_idxes = np.argsort(-agg_score[b])[:sel_num]
            cur_query['sel'].extend([int(i) for i in max_col_idxes])
            cur_query['agg'].extend([i[0] for i in max_agg_idxes])
            cur_query['cond_conn_op'] = np.argmax(where_rela_score[b])
            cur_query['conds'] = []
            cond_num = np.argmax(cond_num_score[b])
            all_toks = ['<BEG>'] + q[b] + ['<END>']
            max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
            for idx in range(cond_num):
                cur_cond = []
                cur_cond.append(max_idxes[idx]) # where-col
                cur_cond.append(np.argmax(cond_op_score[b][idx])) # where-op
                cur_cond_str_toks = []
                for str_score in cond_str_score[b][idx]:
                    str_tok = np.argmax(str_score[:len(all_toks)])
                    str_val = all_toks[str_tok]
                    if str_val == '<END>':
                        break
                    cur_cond_str_toks.append(str_val)
                cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))
                cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)
        return ret_queries
