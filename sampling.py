import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import random
import torch.nn.functional as F

def counter_node_freq(corpus):
    all_nodes = [node for sen in corpus for node in sen]
    nodes_freq_dict =Counter(all_nodes) #node:frequency   
    node_freq_array =np.array([freq for node,freq in nodes_freq_dict.items()])
    nodes_set = [node for node,freq in nodes_freq_dict.items()]
    node_freq_array = node_freq_array**0.75
    node_freq_array = node_freq_array/np.sum(node_freq_array)#

    return node_freq_array, nodes_set
    
def positive_samples(corpus, num_pos=2, num_neg=10):
    nodes_set = list(set([node for sen in corpus for node in sen]))
    # node_freq_array,nodes_set =counter_node_freq(corpus) #use weight
    pos_samps=[]
    neg_samps =[]
    
    for sen in corpus:
        pos_samp =[]
        neg_samp= []
        for idx,node in enumerate(sen):
            context_nodes = sen[max(0,idx-num_pos):idx]+sen[idx+1:min(idx+num_pos+1,len(sen))]
            neg_nodes =random.choices(nodes_set, k=num_neg)
            
            while len(context_nodes)<2*num_pos:
                context_nodes.append(random.choice(context_nodes))
            while len(set(context_nodes)&set(neg_nodes))>0:
                #neg_nodes =random.choices(nodes_set, weights =node_freq_array, k=num_neg)
                neg_nodes =random.choices(nodes_set, k=num_neg)
            pos_samp.append(context_nodes)
            neg_samp.append(neg_nodes)
        pos_samps.append(pos_samp)
        neg_samps.append(neg_samp)
            
    return pos_samps,neg_samps,nodes_set
     
    
class EmbeddingModel(nn.Module):
    def __init__(self, num_nodes, emb_dim):
        super(EmbeddingModel,self).__init__()
        
        self.in_emb = nn.Embedding(num_embeddings=num_nodes,embedding_dim=emb_dim)
        # init_range =0.5/emb_dim
        # self.in_emb.weight.data.uniform_(-init_range,init_range)
        # self.in_emb.weigh.data.xavier_uniform_()
        self.out_emb =nn.Embedding(num_embeddings =num_nodes, embedding_dim =emb_dim)
        # self.out_emb.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, batch, pos_samps, neg_samps):
        '''
        Parameters
        ----------
        batch : [batch]
        pos_samps : [batch,num_pos*2]
        neg_samps : [batch, num_neg*2]

        '''
        batch_emb = self.in_emb(batch)#[batch, emb_dim]
        # print(batch_emb.size())
        pos_emb = self.out_emb(pos_samps)#[batch,num_pos*2, emb_dim]
        neg_emb =self.out_emb(neg_samps)#[batch,num_neg*2, emb_dim]
        # print(pos_emb.size())
        batch_emb = batch_emb.unsqueeze(2) #[batch,emb_dim, 1]
        pos_proba = torch.bmm(pos_emb,batch_emb).squeeze()#[batch, num_pos*2, 1]
        neg_proba = torch.bmm(neg_emb,-batch_emb).squeeze()#[batch, num_neg*2, 1]
        # print(pos_proba.size())
        # print(pos_proba)
        loss_pos = F.logsigmoid(pos_proba).sum(1)
        loss_neg = F.logsigmoid(neg_proba).sum(1)
        return -(loss_pos + loss_neg)
        
    def NodeVec(self):
        return self.in_embed.weight.detach().numpy()        
        

    
    
    
    
    
    
    
    
    
    
    
    