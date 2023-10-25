
import torch
import pandas as pd
import math
import random
import numpy as np
import dgl
import pickle as pkl
import datetime
import copy
import math
from conf import Config
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : the save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
        

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        path = self.save_path
        torch.save(model.state_dict(), path)	# The parameters of the current optimal model will be stored here.
        
        self.val_loss_min = val_loss


def test(model,test_data,data_type,global_graph=None,period_glist=None,early_stopping =None):

    config =Config()
    model.eval()
    #graph aggregation
    model.forward(global_graph,period_glist)
    
    with torch.no_grad ():
        test_labels,predict_labels,imp_indexs =[],[],[]
        test_sample=0

        while True:
            
            data_dict = test_data.next_batch(data_type=data_type)
            predict=model.get_loss(data_dict,global_graph,period_glist,train=False)
            
            sample_num = len(data_dict['USER'])
            test_sample+=sample_num
            

            test_labels+=data_dict['LABEL'].reshape(-1).tolist()
            predict_labels+=predict.reshape(-1).tolist()
            
            key_lst = data_dict['SAMPLE_NUM']
            imp_indexs+=key_lst
            
            if test_data.step>=test_data.total_step:
                break
        print("{}样本数量:{}".format(now_time(),test_sample),end=" ")
        
        test_labels = np.array(test_labels)
        predict_labels = np.array(predict_labels)
        #calculate the metrics for full-period
        if config.pair_metrics is not None:
            g_l, g_p = group_labels(test_labels[:],predict_labels[:], imp_indexs)
            metrics=[]
            if data_type =='valid': #hit@10 of valid data set
                metrics = cal_metric(
                    g_l, g_p, ["hit@10"]
                )
            else:
                metrics = cal_metric(
                    g_l, g_p, config.pair_metrics
                )
            
            for metric,value in metrics.items():
                print(metric,":",value,end=" ")
            print()
        
        if  early_stopping:
            # the hit@10 metric is alternative
            early_stopping(-metrics['hit@10'], model)
            if early_stopping.early_stop:
                print("Early stopping")
                return True 
        return False



        
def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '

class Dataset:
    def __init__(self,config,data,test_batchsize=None) -> None:
        self.device = config.device
        if test_batchsize:
            self.batch_size = test_batchsize
        else:
            self.batch_size = config.batch_size//2 
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.data = data
        self.per_food_num = config.per_food_num
        self.time_period = config.time_period
        self.store_list= list(range(config.store_num))
        self.time_type = config.time_type
        self.time_num = config.time_num
        self.load_u2r_r2i(config.user2store_path,config.store2food_path,config.user2food_path)
        
    def load_u2r_r2i(self,user2store_path,store2food_path,user2food_path):
        self.store2food = pkl.load(open(store2food_path,'rb'))
        self.user2store = pkl.load(open(user2store_path,'rb'))
        self.user2food = pkl.load(open(user2food_path,'rb'))
    def neg_food_store(self,userid,max_count=1000):
        
        count=0
        while True: 
            store_ = random.choice(self.store_list)
            
            if store_ in self.user2store[userid]: #interact with the store 
                continue
            count+=1
            #print(store_,self.store2food.keys())
            food_list = self.store2food[store_]
            
            user_inter_food_flag=False
            for i in food_list:
                if i in self.user2food[userid]:
                    user_inter_food_flag=True
                    break
            if not user_inter_food_flag: # don't interact with the store and the food
                return store_,food_list
            if count>max_count:  #the maxium number of random sampling
                return store_,food_list

    def next_batch(self,data_type='train'):
        
        if self.step == self.total_step:
            self.step = 0
            np.random.seed(0)
            np.random.shuffle(self.index_list)
            
        
        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        user_batch =[]

        store_batch =[]
        neg_store_batch=[]
        foodlst_batch =[]
        neg_foodlst_batch=[]
        label_batch =[]
        
        time_index = []
        
        time_period = self.time_period

        
        hour_test =[]
        time_batch = []
        sample_num_batch =[]
        for idx in self.index_list[start:offset]:
            
            user,store,foodlst,hour,label,sample_num = self.data[idx]
            foodlst = foodlst[:self.per_food_num]+[0]*(self.per_food_num-len(foodlst))
            
            
            
            hour_test.append(hour)
            idx=0
            tmp_ts=[0]*len(time_period)
            
            for j in range(len(time_period)):

                if hour<time_period[j]:
                    idx=j
                    
                    break

            tmp_ts[idx]=1
            time_index.append(tmp_ts) 
            if self.time_type=="period":
                time_batch.append(idx)
            elif self.time_type == 'hour':
                
                time_batch.append(hour)
            
            else:
                raise Exception("invalid time_type {}".format(self.time_type))


            user_batch.append(user)
            
            store_batch.append(store)
            foodlst_batch.append(foodlst)
            label_batch.append(label)
            sample_num_batch.append(sample_num)
            
            
            if data_type=='train': #for train dataset
                neg_store,neg_foodlst=self.neg_food_store(user)
                neg_foodlst =neg_foodlst[:self.per_food_num]+[0]*(self.per_food_num-len(neg_foodlst))
                neg_store_batch.append(neg_store)
                neg_foodlst_batch.append(neg_foodlst)
            

        data_dict={}
        if data_type=='train':
            data_dict['NEG_RES']=torch.tensor(neg_store_batch,dtype=torch.int64).to(self.device)
            data_dict['NEG_ITEMLST']=torch.tensor(neg_foodlst_batch,dtype=torch.int64).to(self.device)
        else:
                
            data_dict['SAMPLE_NUM'] = sample_num_batch
        data_dict["LABEL"]=torch.tensor(label_batch,dtype=torch.float32).unsqueeze(dim=1).to(self.device)
        data_dict['USER']=torch.tensor(user_batch,dtype=torch.int64).to(self.device)
        data_dict["ITEMLST"]=torch.tensor(foodlst_batch,dtype=torch.int64).to(self.device)
        data_dict["RES"]=torch.tensor(store_batch,dtype=torch.int64).to(self.device)
        
        #data_dict['TMASK']= torch.tensor(time_index,dtype=torch.float32).unsqueeze(dim=-1).to(self.device)
        data_dict['TIME'] = torch.tensor(time_batch,dtype=torch.int64).to(self.device)

        return data_dict
 
           

def group_labels(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.
    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.
    Returns:
        all_labels: labels after group.
        all_preds: preds after group.
    """

    all_keys = list(set(group_keys))
    g_l = {k: [] for k in all_keys}
    g_p = {k: [] for k in all_keys}
    for l, p, k in zip(labels, preds, group_keys):
        g_l[k].append(l)
        g_p[k].append(p)

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(np.array(g_l[k]))
        all_preds.append(np.array(g_p[k]))
    

    return all_labels, all_preds





def cal_metric(true_labels, pred_labels,metrics):

    res = {}

 
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        

        order=np.argsort(pred_label)[::-1]
        
        for metric in metrics:
            y_true, y_score =copy.deepcopy(true_label),copy.deepcopy(pred_label)
            if metric == "mrr":
                y_true = np.take(y_true, order)  
                
                rr_score = y_true / (np.arange(len(y_true)) + 1)  
                res['mrr']= res.get('mrr',0)+np.sum(rr_score) / np.sum(y_true)
            elif metric =='auc':

                res['auc']=res.get('auc',0)+roc_auc_score(y_true, y_score) 
            elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            
                hit_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    hit_list = [int(token) for token in ks[1].split(";")]
                for k in hit_list:
                    
                    ground_truth = np.where(y_true == 1.0)[0]
                    
                    argsort = order[:k]
                    #print("hit",ground_truth,argsort,y_score)
                    for idx in argsort:
                        if idx in ground_truth:
                            res["hit@{0}".format(k)]=res.get("hit@{0}".format(k),0)+1
                            break
                    res["hit@{0}".format(k)]=res.get("hit@{0}".format(k),0)+0
            elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            
                ndcg_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    ndcg_list = [int(token) for token in ks[1].split(";")]
                for k in ndcg_list:

    
                    tmp_y_true = np.take(y_true, order[:k])
                
                    gains = 2 ** tmp_y_true - 1  
                    discounts = np.log2(np.arange(len(tmp_y_true)) + 2)
                    actual = np.sum(gains / discounts)

                    order2 = np.argsort(y_true)[::-1]
    
                    tmp_y_true = np.take(y_true, order2[:k])
                
                    gains = 2 ** tmp_y_true - 1  
                    discounts = np.log2(np.arange(len(tmp_y_true)) + 2)
                    best = np.sum(gains / discounts)
                    res["ndcg@{0}".format(k)] = res.get("ndcg@{0}".format(k),0)+actual/best
            
   
    count = len(true_labels)
    for key in res.keys():
        #print(key,res[key])
        res[key]=round(res[key]/count,4)
    return res



def load_train_data(data,store2food_path=None,data_type ='test'):
    store2food = pkl.load(open(store2food_path,'rb'))  
    total_num = len(data)
    train_dataset =[]
    for i in range(total_num):
        user = data['user'][i]
        store = data['res'][i]
        foodlst = store2food[store] #the most frequently clicked food set of store: due to the ground-truth interacted food for valid and test data is unknow. 
        time = data['hour'][i]
        label = data['label'][i]
        sample_num =0
        if data_type!='train':
            sample_num=data['sample_num'][i]
        train_dataset.append([user,store,foodlst,time,label,sample_num])
    return train_dataset
def load_global_train_graph(dataset):
    # construct the full-period graph
    train_graph_triplets ={
        'uo':[],
        'us':[],
        'so':[]
    }
    total_num = len(dataset)
    for i in range(total_num):
        user,store,foodlst,label = dataset['user'][i],dataset['res'][i],eval(dataset['itemlst'][i]),dataset['label'][i]
        if label==0:  #only the positive interaction to construct the graph
            continue
        
        for food in foodlst:
            train_graph_triplets['uo'].append([user,food])
            train_graph_triplets['so'].append([store,food])
        
        train_graph_triplets['us'].append([user,store])

    return train_graph_triplets
def load_time_graph(dataset,time_period):
    
    time_stage_num =len(time_period)
    time_list=[]
    
    for i in range(time_stage_num):
        time_list.append({
        'uo':[],
        'us':[],
        'so':[]
    })
    total_num = len(dataset)
    for i in range(total_num):
        user,store,foodlst,hour,label =dataset['user'][i],dataset['res'][i],eval(dataset['itemlst'][i]),dataset['hour'][i],dataset['label'][i]
        
        if label==0:
            continue
        insertidx=0 
        for j in range(time_stage_num): 
            if hour < time_period[j]:  #[20-5,5-10,10-15,15-20]
                insertidx=j
                break
        if hour>20:
            assert insertidx==0 
        for food in foodlst:
            time_list[insertidx]['uo'].append([user,food])
            
            time_list[insertidx]['so'].append([store,food])
        
        time_list[insertidx]['us'].append([user,store])
        

    return time_list

def load_train_graph(dataset,config):


    
    time_period = config.time_period
    train_global_graph = load_global_train_graph(dataset)
    train_time_graph_lst =[]
    

    train_time_graph_lst = load_time_graph(dataset,time_period)
    
        
    return train_global_graph,train_time_graph_lst


def add_graph2ori(ori_graph,add_graph):

    new_graph= copy.deepcopy(ori_graph)
    
    inter_key=['us','uo','so']
    
    for key in inter_key:
        new_graph[key]+=add_graph[key]
    return new_graph

def load_data(config):
    '''
    Output:
    train/valid/test dataset: ['user','store','foodlst','hour','label','sample_num']
    train_global_graph
    {
        'uo': [[u,i],...]
        'us': [[u,i],...]
        'so': [[r,i],...] 
    }

    train_time_graph_lst: the graph interaction for different periods
    [{'uo': [[u,i],...], 'us': [[u,i],...],'so': [[r,i],...]} *period_num]
    '''
    #
    train_data = pd.read_csv(config.train_data_path)
    valid_data = pd.read_csv(config.valid_data_path)
    test_data = pd.read_csv(config.test_data_path)

    #1.load dataset
    train_dataset = load_train_data(train_data,config.store2food_path,'train')
    valid_dataset = load_train_data(valid_data,config.store2food_path)  
    test_dataset = load_train_data(test_data,config.store2food_path)
    
    #2.construct the train graph
    train_global_graph,train_time_graph_lst= load_train_graph(train_data,config)

    # add the additional interaction in valid dataset for the test data
    add_test_global_graph, add_test_time_graph_lst = load_train_graph(valid_data,config)
    test_global_graph = add_graph2ori(train_global_graph,add_test_global_graph)
    test_time_graph_lst=[]
    for ori_graph,add_graph in zip(train_time_graph_lst,add_test_time_graph_lst):
        test_time_graph_lst.append(add_graph2ori(ori_graph,add_graph))

    return train_dataset, valid_dataset,test_dataset,(train_global_graph,train_time_graph_lst),(test_global_graph,test_time_graph_lst)



def build_graph(num_user,num_store,num_food,train_graph_triplets,device=None):    
    '''
    construct graph
    '''
    ur_u,ur_r = np.array(train_graph_triplets['us']).transpose()
    ui_u,ui_i = np.array(train_graph_triplets['uo']).transpose()
    ri_r,ri_i = np.array(train_graph_triplets['so']).transpose()
    num_nodes_dict={
        'user':num_user,
        'store':num_store,
        'food':num_food
    }
    graph_data = {
        ('user', 'us', 'store'): (ur_u, ur_r),
        ('store', 'su', 'user'): (ur_r, ur_u),
        ('food', 'os', 'store'): (ri_i, ri_r),
        ('user', 'uo', 'food'): (ui_u, ui_i),
        ('food', 'ou', 'user'): (ui_i, ui_u),
        ('store', 'so', 'food'): (ri_r, ri_i),
    }
    g = dgl.heterograph(graph_data,num_nodes_dict=num_nodes_dict)
    

    
    if device:
        g=g.to(device)
    return g

