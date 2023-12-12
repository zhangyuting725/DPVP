
from utils import *
from mygraph.DPVP import *
import torch
from conf import Config
import numpy as np
import random,os


def run_time(config):
    
    print(now_time()+" start loading data")
    
    train_dataset,valid_dataset,test_dataset,train_graph_triplets,test_graph_triplets = load_data(config)
    global_graph_triplets,period_graph_triplets = train_graph_triplets

    # for train data:
    print(now_time()+" start building global graph")
    global_graph = build_graph(config.user_num,config.store_num,config.food_num, global_graph_triplets,config.device)
    print(now_time()+" start building time graph list")
    period_glist = [build_graph(config.user_num,config.store_num,config.food_num, time_graph,config.device) for time_graph in period_graph_triplets]
    
    # for valid data:
    valid_global_graph= global_graph
    valid_period_glist = period_glist

    # for test data: add the interactions in validation dataset into the graph
    test_global_graph_triplets,test_period_graph_triplets =test_graph_triplets
    test_global_graph = build_graph(config.user_num,config.store_num,config.food_num, test_global_graph_triplets,config.device)
    test_period_glist = [build_graph(config.user_num,config.store_num,config.food_num, time_graph,config.device) for time_graph in test_period_graph_triplets]
    
    model = DPVP(config).to(config.device)
    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)
    
    model_path = config.save_model_dir+"/main.pt"
    print("-------------Save path----------",model_path)
    
    valid_data = Dataset(config,valid_dataset,test_batchsize=config.test_batchsize)
    test_data = Dataset(config,test_dataset,test_batchsize=config.test_batchsize)
    


    early_stopping = EarlyStopping(model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_data = Dataset(config,train_dataset)
    print(now_time()+" start training")
    for epoch in range(1,config.epochs+1):
        print("-"*20)
        train_sample=0
        train_loss_sum=0
        
        while True:

            data_dict = train_data.next_batch(data_type='train')

            sample_num = len(data_dict['USER'])
            train_sample+=sample_num
            loss = model.get_loss(data_dict,global_graph,period_glist)  
            optimizer.zero_grad()
            loss.backward()
            train_loss_sum+=loss.item()*sample_num
            
            optimizer.step()
            if train_data.step>=train_data.total_step:
                break
        print("{} epoch:{},样本数量:{},训练loss:{:.4f}".format(now_time(),epoch,train_sample,train_loss_sum/train_sample))   

        if epoch %1==0:
            print("验证",end=" ")
            flag=test(model,valid_data,'valid',valid_global_graph,period_glist=valid_period_glist,early_stopping=early_stopping)
            if epoch%10==0:
                print("测试",end=" ")
                test(model,test_data,'test',test_global_graph,period_glist=test_period_glist)
            if flag:
                break
    
    model.load_state_dict(torch.load(model_path))
    test(model,test_data,'test',test_global_graph,period_glist=test_period_glist)
    os.remove(model_path)
        
if __name__=='__main__':
    config= Config()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    print("------------the parameter setting-------------")
    print(config)
    run_time(config)
         


        