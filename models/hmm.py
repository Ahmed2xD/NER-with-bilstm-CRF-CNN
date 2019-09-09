import torch


class HMM(object):
    def __init__(self, N, M):
       
        self.N = N
        self.M = M

        self.A = torch.zeros(N, N)
        self.B = torch.zeros(N, M)
        self.Pi = torch.zeros(N)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        assert len(tag_lists) == len(word_lists)
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i+1]]
                self.A[current_tagid][next_tagid] += 1
       
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 估计初始状态概率
        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def test(self, word_lists, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.decoding(word_list, word2id, tag2id)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists

    def decoding(self, word_list, word2id, tag2id):
   
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        
        seq_len = len(word_list)
        viterbi = torch.zeros(self.N, seq_len)
       
        backpointer = torch.zeros(self.N, seq_len).long()

       
        start_wordid = word2id.get(word_list[0], None)
        Bt = B.t()
        if start_wordid is None:
            bt = torch.log(torch.ones(self.N) / self.N)
        else:
            bt = Bt[start_wordid]
        viterbi[:, 0] = Pi + bt
        backpointer[:, 0] = -1

        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)
            if wordid is None:
                bt = torch.log(torch.ones(self.N) / self.N)
            else:
                bt = Bt[wordid]  
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(
                    viterbi[:, step-1] + A[:, tag_id],
                    dim=0
                )
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        best_path_prob, best_path_pointer = torch.max(
            viterbi[:, seq_len-1], dim=0
        )

        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list
