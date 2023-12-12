
import torch
import torch.nn as nn
from mygraph.BaseP import *
  
class DPVP(nn.Module):
    def __init__(self,config) -> None:
        super(DPVP, self).__init__()
        self.config = config
        emb_dim = config.emb_dim
        layer_num = config.layer_num
        self.init_weight(config)
        self.storelayers = nn.ModuleList()
        for _ in range(layer_num):
            self.storelayers.append(MultiGraphlayer(config,'store'))
        self.foodlayers = nn.ModuleList()
        for _ in range(layer_num):
            self.foodlayers.append(MultiGraphlayer(config,'food'))

        self.logsigmoid = nn.LogSigmoid()
        input_size = emb_dim*8
        self.score_predictor= nn.Sequential(
                nn.Linear(input_size,input_size//2),
                nn.LeakyReLU(),
                nn.Linear(input_size//2,input_size//4),
                nn.LeakyReLU(),
                nn.Linear(input_size//4,1)
        )
        
        self.linear_K= nn.Linear(emb_dim,emb_dim)
        self.food_attn = Attn(config.attn_method,config.emb_dim)
        
        
        self.user_o_linear = nn.Linear(config.emb_dim*2,config.emb_dim)
        self.user_o_time_attn = Attn(config.attn_method,config.emb_dim)
        self.food_o_linear = nn.Linear(config.emb_dim*2,config.emb_dim)
        self.food_o_time_attn = Attn(config.attn_method,config.emb_dim)
        self.store_s_linear = nn.Linear(config.emb_dim*2,config.emb_dim)
        self.store_s_time_attn = Attn(config.attn_method,config.emb_dim)
        self.user_s_linear = nn.Linear(config.emb_dim*2,config.emb_dim)
        self.user_s_time_attn = Attn(config.attn_method,config.emb_dim)
        
            
        

    def init_weight(self,config):
        user_num = config.user_num
        store_num = config.store_num
        food_num = config.food_num

        emb_dim = config.emb_dim
        time_num = config.time_num

        self.time_embedding = nn.Embedding(time_num, emb_dim)
        initializer = nn.init.xavier_uniform_
        
        self.embedding_dict = nn.ParameterDict({
            'user_emb_s': nn.Parameter(initializer(torch.empty(user_num, emb_dim))),
            'food_emb_s': nn.Parameter(initializer(torch.empty(food_num, emb_dim))),
            'store_emb_s': nn.Parameter(initializer(torch.empty(store_num, emb_dim))),
            'user_emb_o': nn.Parameter(initializer(torch.empty(user_num, emb_dim))),
            'food_emb_o': nn.Parameter(initializer(torch.empty(food_num, emb_dim))),
            'store_emb_o': nn.Parameter(initializer(torch.empty(store_num, emb_dim))),
        })
        
    def one_graph_train(self,train_graph,embedding_dict):
        #---------aggregate a single graph------
        #1. embedding init:
        # store-level embedding
        prev_user_embedding_s = embedding_dict['user_emb_s']
        prev_food_embedding_s = embedding_dict['food_emb_s']
        prev_store_embedding_s = embedding_dict['store_emb_s']

        # food-level embedding
        prev_user_embedding_o = embedding_dict['user_emb_o']
        prev_food_embedding_o = embedding_dict['food_emb_o']
        prev_store_embedding_o = embedding_dict['store_emb_o']
        
        #2. aggregate
        for  i, layer in enumerate(self.storelayers):
            if i==0:
                user_emb_s, store_emb_s = layer(train_graph,prev_user_embedding_s,prev_store_embedding_s,prev_food_embedding_s)
            else:
                user_emb_s, store_emb_s = layer(train_graph,user_emb_s,store_emb_s,prev_food_embedding_s)  
            prev_user_embedding_s = prev_user_embedding_s+ user_emb_s*(1/(i+2))
            prev_store_embedding_s = prev_store_embedding_s+ store_emb_s*(1/(i+2))
        for  i, layer in enumerate(self.foodlayers):
            if i==0:
                user_emb_o, food_emb_o = layer(train_graph,prev_user_embedding_o,prev_store_embedding_o,prev_food_embedding_o)
            else:
                user_emb_o, food_emb_o = layer(train_graph,user_emb_o,prev_store_embedding_o,food_emb_o)  #注意这里的food,embeddinguser_emb_o, food_emb_o = layer(train_graph,user_emb_o,food_emb_o,prev_store_embedding_o,query_embedding)  #注意这里的food,embedding
            prev_user_embedding_o= prev_user_embedding_o+ user_emb_o*(1/(i+2))
            prev_food_embedding_o = prev_food_embedding_o+ food_emb_o*(1/(i+2))
        return prev_user_embedding_s,prev_store_embedding_s,prev_user_embedding_o,prev_food_embedding_o
    def forward(self,global_graph,time_graph_lst):
        #obtain multiple repstoreentations in multiple graphs 
        '''
        Input
        global_graph: full-period graph
        time_graph_lst :T graphs

        Output
        glst_embs  #[T,4,dim]
        [
            [[],[],[],[]],  #user_emb_s, store_emb_s, user_emb_o, food_emb_o
            [[],[],[],[]]

        ]
        global_emb #[4,dim]
        [[],[],[],[]]
        '''
        self.glst_embeddings= []
        for g in time_graph_lst:
            user_embedding_s,store_embedding_s,user_embedding_o, food_embedding_o = self.one_graph_train(g,self.embedding_dict)
            self.glst_embeddings.append([user_embedding_s,store_embedding_s,user_embedding_o, food_embedding_o])
        user_embedding_s,store_embedding_s,user_embedding_o, food_embedding_o=self.one_graph_train(global_graph,self.embedding_dict)
        self.global_embs =[user_embedding_s,store_embedding_s,user_embedding_o, food_embedding_o]
    def get_attn_food_emb(self,food_embeddings,food_lst,sample_user_emb_o):
        '''
        food_emb: [N,per_food_num,dim]
        sample_user_emb_o: [N,dim]
        '''
        #------get user-aware repstoreentation of foodlist
        food_emb = food_embeddings[food_lst]
        ones = torch.ones(food_lst.shape).to(self.config.device)
        zeros = torch.zeros(food_lst.shape).to(self.config.device)
        mask = torch.where(food_lst>0,ones,zeros)  #N,M
        
            
        key = self.linear_K(sample_user_emb_o) #[N,dim]
        
        food_emb_o,_= self.food_attn(key,food_emb,mask=mask)     #[N,dim]  
        return food_emb_o
        

    


    
    def emb_aggr(self,data_dict,neg=False):
        
        #--- time gate------
        time_emb = self.time_embedding(data_dict['TIME']) #[N,time_num,]
        #1.global graph 
        global_user_emb_s = self.global_embs[0][data_dict['USER']] #store-level
        global_user_emb_o = self.global_embs[2][data_dict['USER']] #food-level
        if neg:
            global_store_emb_s = self.global_embs[1][data_dict['NEG_RES']]
            global_food_emb_o = self.get_attn_food_emb(self.global_embs[3],data_dict['NEG_ITEMLST'],global_user_emb_o) 
        else:
            global_store_emb_s = self.global_embs[1][data_dict['RES']]
            global_food_emb_o = self.get_attn_food_emb(self.global_embs[3],data_dict['ITEMLST'],global_user_emb_o) 
        
        #2.period graph 
        user_emb_sl = []  #[T,N,dim]
        store_emb_sl =[]
        user_emb_ol=[]
        food_emb_ol=[]

        for i in range(len(self.glst_embeddings)):
            user_emb_s =self.glst_embeddings[i][0][data_dict['USER']]
            user_emb_o = self.glst_embeddings[i][2][data_dict['USER']]
            
            if neg:
                store_emb_s = self.glst_embeddings[i][1][data_dict['NEG_RES']]
                food_emb_o = self.get_attn_food_emb(self.glst_embeddings[i][3],data_dict['NEG_ITEMLST'],user_emb_o) 
            else:

                store_emb_s = self.glst_embeddings[i][1][data_dict['RES']]
                food_emb_o = self.get_attn_food_emb(self.glst_embeddings[i][3],data_dict['ITEMLST'],user_emb_o) 

            user_emb_sl.append(user_emb_s)
            store_emb_sl.append(store_emb_s)
            user_emb_ol.append(user_emb_o)
            food_emb_ol.append(food_emb_o)
        user_emb_sl = torch.stack(user_emb_sl,0).transpose(0,1) #T,N,dim --> N,T,dim
        store_emb_sl = torch.stack(store_emb_sl,0).transpose(0,1)
        user_emb_ol = torch.stack(user_emb_ol,0).transpose(0,1)
        food_emb_ol = torch.stack(food_emb_ol,0).transpose(0,1)

        #3.time gate
        user_emb_s = self.user_s_linear(torch.concat([global_user_emb_s,time_emb],1))
        user_emb_s,_ = self.user_s_time_attn(user_emb_s,user_emb_sl)

        store_emb_s = self.store_s_linear(torch.concat([global_store_emb_s,time_emb],1))
        store_emb_s,_ = self.store_s_time_attn(store_emb_s,store_emb_sl)

        user_emb_o = self.user_o_linear(torch.concat([global_user_emb_o,time_emb],1))
        user_emb_o,_ = self.user_o_time_attn(user_emb_o,user_emb_ol)
        
        food_emb_o = self.food_o_linear(torch.concat([global_food_emb_o,time_emb],1))  
        food_emb_o,_ = self.food_o_time_attn(food_emb_o,food_emb_ol)
        
        #4.  dual period-varying repstoreentations
        sample_user_emb_s = torch.cat([global_user_emb_s,user_emb_s],dim=1)
        sample_store_emb_s = torch.cat([global_store_emb_s,store_emb_s],dim=1)
        sample_user_emb_o = torch.cat([global_user_emb_o,user_emb_o],dim=1)
        sample_food_emb_o = torch.cat([global_food_emb_o,food_emb_o],dim=1)
        return sample_user_emb_s,sample_store_emb_s,sample_user_emb_o,sample_food_emb_o
    
    def get_loss(self,data_dict,global_graph=None,time_graph_lst=None,train=True):
        # graph aggr
        if train:
            self.forward(global_graph,time_graph_lst)
            
        sample_user_emb_s,pos_store_emb_s,sample_user_emb_o,pos_food_emb_o = self.emb_aggr(data_dict,neg=False)

        inputs = torch.cat([sample_user_emb_s,pos_store_emb_s,sample_user_emb_o,pos_food_emb_o],dim=1)
        
        pos_score = self.score_predictor(inputs)

        if train:
            sample_user_emb_s,neg_store_emb_s,sample_user_emb_o,neg_food_emb_o = self.emb_aggr(data_dict,neg=True)
            inputs = torch.cat([sample_user_emb_s,neg_store_emb_s,sample_user_emb_o,neg_food_emb_o],dim=1)
            
            neg_score = self.score_predictor(inputs)
            loss = -torch.mean(self.logsigmoid(pos_score-neg_score))
            return loss
        else:
            return pos_score
       


        


    
    
    