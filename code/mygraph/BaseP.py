import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

class Attn(torch.nn.Module):
    
    def __init__(self, method, hidden_size,q_size=None):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat','mlp']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.linear = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            
            initscale=0.05
            self.linear = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(nn.init.uniform_(torch.FloatTensor(hidden_size),a=-initscale,b=initscale))
        elif self.method == 'mlp':
            input_size = q_size+hidden_size
            self.mlp= nn.Sequential(
                nn.Linear(input_size,input_size//2),
                nn.LeakyReLU(),
                nn.Linear(input_size//2,input_size//4),
                nn.LeakyReLU(),
                nn.Linear(input_size//4,1),
                nn.LeakyReLU(),
            )
            

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)  #[N,dim]*[T,N,dim]

    def general_score(self, hidden, encoder_output):
        energy = self.linear(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.linear(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()

        return torch.sum(self.v * energy, dim=2)  #[T,N]
    def mlp_score(self,hidden,encoder_output):
        T = encoder_output.shape[0]
        dim = encoder_output.shape[-1]
        #print("a",hidden[:10])
        hidden = hidden.repeat([T,1])  #[T*N,q_size]
        encoder_output = encoder_output.reshape(-1,dim)  #[T*N,dim]
        #print("c",encoder_output[:10])
        x=torch.cat([hidden,encoder_output],dim=1)
        
        
        return self.mlp(x).reshape(T,-1)  #T,N
        


    def forward(self, hidden, ori_encoder_outputs,mask=None):
        '''
        input: hidden: [N,dim]
        ori_encoder_output: [N,T,dim]
        mask: [N,T]
        output:N,T
        '''
        encoder_outputs = ori_encoder_outputs.transpose(0,1)  #[T,N,dim]
        #print(encoder_outputs.shape)
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)#[T,N]
        elif self.method == 'mlp':
            attn_energies = self.mlp_score(hidden, encoder_outputs)#[T,N]
        #return attn_energies
        # 转置max_length和batch_size维度
        
        attn_energies = attn_energies.t()  #[N,T]
        
        if mask!=None:
            A=F.softmax(attn_energies, dim=1) #N,T
            
            A = A*mask #N,T
            A_sum=torch.sum(A, dim=1) #N
            threshold=torch.ones_like(A_sum)*1e-5 
            A_sum = torch.max(A_sum, threshold).unsqueeze(1) #[N,1]
            
            A = A / A_sum #[N,T]
            attn_energies =A.unsqueeze(1) #[N,1,T]

        # 返回softmax归一化概率分数（增加维度）
        else:
            attn_energies=F.softmax(attn_energies, dim=1).unsqueeze(1)  #[N,1,T]
        
        context = attn_energies.bmm(ori_encoder_outputs) #[N,1,T]*[N,T,dim]
        
        return context.squeeze(1),attn_energies.squeeze(1) #[N,dim]

    

class MultiGraphlayer(nn.Module):
    def __init__(self,config,ctype):
        super(MultiGraphlayer, self).__init__()
        self.aggr_type = config.aggr_type
        self.aggr_method = config.aggr_method
        self.c_type = ctype
        

    def forward(self, g, user_emb,store_emb,food_emb): 
        with g.local_scope():
           
            #-------普通处理的方式  mean h+query 
            g.nodes['user'].data['h']=user_emb
            g.nodes['store'].data['h']=store_emb
            g.nodes['food'].data['h']=food_emb
            
            if self.c_type == 'food':
                dic= {
                    'uo': (fn.copy_u('h','m'),fn.mean('m','h')),
                    'ou': (fn.copy_u('h','m'),fn.mean('m','h')),
                    'so': (fn.copy_u('h','m'),fn.mean('m','h')),    
                    'os': (fn.copy_u('h','m'),fn.mean('m','h'))
                }
            else:
                dic= {
                    'us': (fn.copy_u('h','m'),fn.mean('m','h')),
                    'su': (fn.copy_u('h','m'),fn.mean('m','h')),
                    'so': (fn.copy_u('h','m'),fn.mean('m','h')),    
                    'os': (fn.copy_u('h','m'),fn.mean('m','h'))
                }
            

            

            g.multi_update_all( 
                dic,
                self.aggr_type

            )
            return g.nodes['user'].data['h'],g.nodes[self.c_type].data['h']
            
