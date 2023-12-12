import argparse
import inspect
import torch



class Config:
    test=True
    dir = "../"

        
    data_size='MT-small'
    epochs = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 1024

    dist_url ='1'
    gpu_count='2'
    
    emb_dim=100
    layer_num=2
    if device ==torch.device('cpu'):
        print("you use cpu now!!!")
    random_seed=0
    
    attn_method = 'general'
    time_attn_method ='general'
    
    

    test_batchsize= 20000


    
    
    lr=1e-4


    
    
    per_food_num = 3 
    
    

    
    time_type = "hour"
    if time_type=="period":
        time_num = 4
    elif time_type=="hour":
        time_num=24
   
    
    
    time_period =[5,10,15,20]
    
    weight_decay=1e-5
    

    aggr_type ="mean"
    aggr_method ="4mean"
    
    
    save_model_dir =dir+"save_model/"+"/"+data_size+"/"
    data_dir = dir+'/'+data_size+"/"

    train_data_path =data_dir+'train_data.csv'
    valid_data_path = data_dir+'valid_data.csv'
    test_data_path = data_dir+'test_data.csv'
    store2food_path = data_dir+'res2item.pkl'
    user2store_path = data_dir+'user2res.pkl'
    user2food_path = data_dir+'user2item.pkl'
    
    
    pair_metrics =["hit@10","ndcg@10","mrr","auc","avg_pos"]



    
    
    user_num=56887
    food_num=5952
    store_num=4059
    word_num=5000#4736 #5000
    per_food_num = 10


    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
