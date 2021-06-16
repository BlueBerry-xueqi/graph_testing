
import collections
import numpy as np
import pandas as pd
from torch.nn.modules import batchnorm
from tqdm import tqdm


## deep gauge
class kmnc(object):
    def __init__(self,train,model, layers,k_bins=1000):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.layers=layers
        self.model = model
        self.k_bins=k_bins
        
        self.upper=[]
        self.lower=[]
        self.upper_gl=[]
        self.lower_gl=[]
        _, _, outputs, batch  = model.extract_intermediate_outputs(train)
        datalength = len(train.dataset)
        for l in layers:
            # print( len(outputs[l]) )
            # print( datalength )
            if len(outputs[l]) != datalength:
                temp =outputs[l]
                self.upper_gl.append(np.max(temp,axis=0))
                self.lower_gl.append(np.min(temp,axis=0))
            else:
                temp=outputs[l].reshape(datalength, -1)
                self.upper.append(np.max(temp,axis=0))
                self.lower.append(np.min(temp,axis=0))

        self.upper=np.concatenate(self.upper,axis=0)
        self.lower=np.concatenate(self.lower,axis=0)
        self.upper_gl=np.concatenate(self.upper_gl,axis=0)
        self.lower_gl=np.concatenate(self.lower_gl,axis=0)
        self.neuron_num=self.upper.shape[0] + self.upper_gl.shape[0]
       


    def fit(self,test):
        '''
        test:测试集数据
        输出测试集的覆盖率
        '''
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, _  = self.model.extract_intermediate_outputs(test)
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l])  != datalength:
                temp = outputs[l]
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                self.neuron_activate_dense.append(temp.copy())
        self.neuron_activate_dense=np.concatenate(self.neuron_activate_dense,axis=1)
        self.neuron_activate_gl=np.concatenate(self.neuron_activate_gl,axis=1)
        act_num=0
        for index in range(len(self.upper)):
                bins=np.linspace(self.lower[index],self.upper[index],self.k_bins)
                act_num+=len(np.unique(np.digitize(self.neuron_activate_dense[:,index],bins)))
        for index in range(len(self.upper_gl)):
                bins=np.linspace(self.lower_gl[index],self.upper_gl[index],self.k_bins)
                act_num+=len(np.unique(np.digitize(self.neuron_activate_gl[:,index],bins)))

        return act_num/float(self.k_bins*self.neuron_num)

    def rank_fast(self,test):
        '''
        test:测试集数据
        输出排序情况
        '''
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, batch  = self.model.extract_intermediate_outputs(test)
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l]) != datalength:
                temp=outputs[l]
                print(temp.shape)
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                print(temp.shape)
                self.neuron_activate_dense.append(temp.copy())
        self.neuron_activate_dense=np.concatenate(self.neuron_activate_dense,axis=1)
        print(self.neuron_activate_dense.shape)
        self.neuron_activate_gl=np.concatenate(self.neuron_activate_gl,axis=1)
        print(self.neuron_activate_gl.shape)

        big_bins=np.zeros((datalength,self.neuron_num,self.k_bins+1))
        for n_index,neuron_activate in tqdm(enumerate(self.neuron_activate_dense)):
            for index in range(len(neuron_activate)):
                bins=np.linspace(self.lower[index],self.upper[index],self.k_bins)
                temp=np.digitize(neuron_activate[index],bins)
                big_bins[n_index][index][temp]=1
        #print(np.max(batch))
        for n_index,neuron_activate in tqdm(enumerate(self.neuron_activate_gl)):
            # n_index = self.upper_gl.shape[0] + n_index
            for index in range(len(neuron_activate)):
                bins=np.linspace(self.lower_gl[index],self.lower_gl[index],self.k_bins)
                temp=np.digitize(neuron_activate[index],bins)
               # print(f"{batch[n_index]}, {self.upper.shape[0] +index}, {temp}")
                big_bins[batch[n_index]][self.upper.shape[0] +index][temp]=1

        big_bins=big_bins.astype('int')
        score_each = big_bins.sum(axis=2).sum(axis=1)
        subset=[]
        lst=list(range(datalength))
        initial=np.random.choice(range(datalength))
        lst.remove(initial)
        subset.append(initial)
        max_cover_num=(big_bins[initial]>0).sum()
        cover_last=big_bins[initial]
        while True:
            flag=False
            for index in tqdm(lst):
                temp1=np.bitwise_or(cover_last,big_bins[index])
                now_cover_num=(temp1>0).sum()
                if now_cover_num>max_cover_num:
                    max_cover_num=now_cover_num
                    max_index=index
                    max_cover=temp1
                    flag=True
            cover_last=max_cover
            if not flag or len(lst)==1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            print(max_cover_num)
        return subset, score_each




class nbc(object):
    def __init__(self,train,model, layers,std=0):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.std=std
        self.model = model
        self.upper=[]
        self.lower=[]
        self.upper_gl=[]
        self.lower_gl=[]
        _, _, outputs, _  = model.extract_intermediate_outputs(train)
        datalength = len(train.dataset)
        for l in layers:
            if len(outputs[l]) != datalength:
                temp =outputs[l]
                self.upper_gl.append(np.max(temp,axis=0)+std*np.std(temp,axis=0))
                self.lower_gl.append(np.min(temp,axis=0)-std*np.std(temp,axis=0))
            else:
                temp=outputs[l].reshape(datalength, -1)
                self.upper.append(np.max(temp,axis=0)+std*np.std(temp,axis=0))
                self.lower.append(np.min(temp,axis=0)-std*np.std(temp,axis=0))
            
        self.upper=np.concatenate(self.upper,axis=0)
        self.lower=np.concatenate(self.lower,axis=0)
        self.upper_gl=np.concatenate(self.upper_gl,axis=0)
        self.lower_gl=np.concatenate(self.lower_gl,axis=0)
        self.neuron_num = self.upper.shape[0] + self.upper_gl.shape[0]
        

    def fit(self,test,use_lower=False):
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, _  = self.model.extract_intermediate_outputs(test)
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l])  != datalength:
                temp = outputs[l]
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                self.neuron_activate_dense.append(temp.copy())
        self.neuron_activate_dense=np.concatenate(self.neuron_activate_dense,axis=1)
        self.neuron_activate_gl=np.concatenate(self.neuron_activate_gl,axis=1)
        act_num=0
        act_num+=(np.sum(self.neuron_activate_dense>self.upper,axis=0)>0).sum()
        act_num+=(np.sum(self.neuron_activate_gl>self.upper_gl,axis=0)>0).sum()
        if use_lower:
            act_num+=(np.sum(self.neuron_activate_dense<self.lower,axis=0)>0).sum()
            act_num+=(np.sum(self.neuron_activate_gl<self.lower_gl,axis=0)>0).sum()

        if use_lower:
            return act_num/(2*float(self.neuron_num))
        else:
            return act_num/float(self.neuron_num)


    def rank_fast(self,test,use_lower=False):
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, batch  = self.model.extract_intermediate_outputs(test)
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l]) != datalength:
                temp=outputs[l]
                print(temp.shape)
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                print(temp.shape)
                self.neuron_activate_dense.append(temp.copy())
        self.neuron_activate_dense=np.concatenate(self.neuron_activate_dense,axis=1)
        print(self.neuron_activate_dense.shape)
        self.neuron_activate_gl=np.concatenate(self.neuron_activate_gl,axis=1)
        print(self.neuron_activate_gl.shape)

        upper=(self.neuron_activate_dense>self.upper)
        lower=(self.neuron_activate_dense<self.lower)
        from torch_scatter import scatter
        import torch
        size = int(np.max(batch) + 1)
        upper_gl=(self.neuron_activate_gl>self.upper_gl)
        lower_gl=(self.neuron_activate_gl<self.lower_gl)
        upper_gl = scatter(torch.tensor(upper_gl), torch.tensor(batch), dim=0, dim_size=size, reduce='add').numpy()
        lower_gl = scatter(torch.tensor(lower_gl), torch.tensor(batch), dim=0, dim_size=size, reduce='add').numpy()
        print(upper_gl.shape)
        upper = np.concatenate(( upper, upper_gl), axis=1)
        lower = np.concatenate(( lower, lower_gl), axis=1)
        score = np.sum(self.neuron_activate_dense>self.upper,axis=1)+np.sum(self.neuron_activate_dense<self.lower,axis=1)

        subset=[]
        lst=list(range(len(test)))
        initial=np.random.choice(range(len(test)))
        lst.remove(initial)
        subset.append(initial)
        max_cover_num=np.sum(upper[initial])
        if use_lower:
            max_cover_num+=np.sum(lower[initial])
        cover_last_1=upper[initial]
        if use_lower:
            cover_last_2=lower[initial]
        while True:
            flag=False
            for index in tqdm(lst):
                temp1=np.bitwise_or(cover_last_1,upper[index])
                cover1=np.sum(temp1)
                if use_lower:
                    temp2=np.bitwise_or(cover_last_2,lower[index])
                    cover1+=np.sum(temp2)
                if cover1>max_cover_num:
                    max_cover_num=cover1
                    max_index=index
                    flag=True
                    max_cover1=temp1
                    if use_lower:
                        max_cover2=temp2
            if not flag or len(lst)==1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            cover_last_1=max_cover1
            if use_lower:
                cover_last_2=max_cover2
            print(max_cover_num)
        return subset, score

    def rank_2(self,test,use_lower=False):
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, batch  = self.model.extract_intermediate_outputs(test)
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l]) != datalength:
                temp=outputs[l]
                print(temp.shape)
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                print(temp.shape)
                self.neuron_activate_dense.append(temp.copy())
        self.neuron_activate_dense=np.concatenate(self.neuron_activate_dense,axis=1)
        print(self.neuron_activate_dense.shape)
        self.neuron_activate_gl=np.concatenate(self.neuron_activate_gl,axis=1)
        print(self.neuron_activate_gl.shape)

        upper=(self.neuron_activate_dense>self.upper)
        lower=(self.neuron_activate_dense<self.lower)
        from torch_scatter import scatter
        import torch
        size = int(np.max(batch) + 1)
        upper_gl=(self.neuron_activate_gl>self.upper_gl)
        lower_gl=(self.neuron_activate_gl<self.lower_gl)
        upper_gl = scatter(torch.tensor(upper_gl), torch.tensor(batch), dim=0, dim_size=size, reduce='add').numpy()
        lower_gl = scatter(torch.tensor(lower_gl), torch.tensor(batch), dim=0, dim_size=size, reduce='add').numpy()
        upper = np.concatenate(( upper, upper_gl), axis=1)
        lower = np.concatenate(( lower, lower_gl), axis=1)
        if use_lower:
            return np.argsort(np.sum(upper,axis=1)+np.sum(lower,axis=1))[::-1]
        else:
            return np.argsort(np.sum(upper,axis=1))[::-1]



class tknc(object):
    def __init__(self,test,model,layers,k=2):
        self.train=test
        self.model=model
        self.layers=layers
        self.k=k
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, self.batch  = self.model.extract_intermediate_outputs(test)
        
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l])  != datalength:
                temp = outputs[l]
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                self.neuron_activate_dense.append(temp.copy())

        self.neuron_num=np.concatenate(self.neuron_activate_dense,axis=1).shape[-1] + np.concatenate(self.neuron_activate_gl,axis=1).shape[-1]
     

    def fit(self,choice_index):
        neuron_activate=0
        for neu in self.neuron_activate_dense:
            temp=neu[choice_index]
            neuron_activate+=len(np.unique(np.argsort(temp,axis=1)[:,-self.k:]))

        for neu in self.neuron_activate_gl:
            for i in choice_index:
                #print(type(self.batch))
                #print(np.sum(self.batch==i))
                temp=neu[ np.nonzero(self.batch==i)[0]]
                neuron_activate+=len(np.unique(np.argsort(temp,axis=1)[:,-self.k:]))
        return neuron_activate/float(self.neuron_num)

    def rank(self,test):
        neuron=[]
        layers_num=0
        datalength = len(test.dataset)
        for neu in self.neuron_activate_dense:
            neuron.append(np.argsort(neu,axis=1)[:,-self.k:]+layers_num)
            layers_num+=neu.shape[-1]
        
        neuron=np.concatenate(neuron,axis=1)
        neuron_gl = collections.defaultdict(list)
        for neu in self.neuron_activate_gl:
            for i in range(datalength):
                temp=neu[self.batch==i]
                neuron_gl[i].extend((np.argsort(temp,axis=1)[:,-self.k:]+layers_num).flatten().tolist())
            layers_num+=neu.shape[-1]

        subset=[]
        lst=list(range(datalength))
        initial=np.random.choice(range(datalength))
        lst.remove(initial)
        subset.append(initial)
        max_cover=len(np.unique(neuron[initial]))

        cover_now=neuron[initial]
        scroe = []
        for i in lst:
            scroe.append( self.fit([i]) )
        while True:
            flag=False
            for index in tqdm(lst):
                temp=np.union1d(cover_now.flatten().tolist(),neuron[index].flatten().tolist()+neuron_gl[index])
                cover1=len(temp)
                if cover1>max_cover:
                    max_cover=cover1
                    max_index=index
                    flag=True
                    max_cover_now=temp
            if not flag or len(lst)==1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            cover_now=max_cover_now
            print(max_cover)
        return subset, scroe

## deepxplore
class nac(object):
    def __init__(self,test,model,layers,t=0):
        self.train=test
        self.model=model
        self.layers=layers
        self.t=t
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, self.batch  = self.model.extract_intermediate_outputs(test)
        
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l])  != datalength:
                temp = outputs[l]
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                self.neuron_activate_dense.append(temp.copy())

        self.neuron_num=np.concatenate(self.neuron_activate_dense,axis=1).shape[-1] + np.concatenate(self.neuron_activate_gl,axis=1).shape[-1]
     


    def fit(self):
        neuron_activate=0
        for neu in self.neuron_activate_dense:
            neuron_activate+=np.sum(np.sum(neu>self.t,axis=0)>0)
        for neu in self.neuron_activate_gl:
            neuron_activate+=np.sum(np.sum(neu>self.t,axis=0)>0)
        return neuron_activate/float(self.neuron_num)

    def rank_fast(self,test):
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, batch  = self.model.extract_intermediate_outputs(test)
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l])  != datalength:
                temp = outputs[l]
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                self.neuron_activate_dense.append(temp.copy())
        self.neuron_activate_dense=np.concatenate(self.neuron_activate_dense,axis=1)
        self.neuron_activate_gl=np.concatenate(self.neuron_activate_gl,axis=1)
        neuron_num= self.neuron_activate_dense.shape[-1] + self.neuron_activate_gl.shape[-1]
        upper=(self.neuron_activate_dense>self.t)
        upper_gl=(self.neuron_activate_gl>self.t)
        from torch_scatter import scatter
        import torch
        size = int(np.max(batch) + 1)
        upper_gl = scatter(torch.tensor(upper_gl), torch.tensor(batch), dim=0, dim_size=size, reduce='add').numpy()
        upper_gl = upper_gl > 0
        upper = np.concatenate(( upper, upper_gl), axis=1)
        score = np.sum(upper, axis=1)/neuron_num
        subset=[]
        lst=list(range(len(test)))
        initial=np.random.choice(range(len(test)))
        lst.remove(initial)
        subset.append(initial)
        max_cover_num=np.sum(upper[initial])
        cover_last_1=upper[initial]
        while True:
            flag=False
            for index in tqdm(lst):
                temp1=np.bitwise_or(cover_last_1,upper[index])
                cover1=np.sum(temp1)
                if cover1>max_cover_num:
                    max_cover_num=cover1
                    max_index=index
                    flag=True
                    max_cover1=temp1
            if not flag or len(lst)==1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            cover_last_1=max_cover1
            print(max_cover_num)
        return subset, score

    def rank_2(self,test):
        self.neuron_activate_dense =[]
        self.neuron_activate_gl =[]
        _, _, outputs, batch  = self.model.extract_intermediate_outputs(test)
        datalength = len(test.dataset)
        for l in self.layers:
            if len(outputs[l])  != datalength:
                temp = outputs[l]
                self.neuron_activate_gl.append(temp.copy())
            else:
                temp=outputs[l].reshape(datalength,-1)
                self.neuron_activate_dense.append(temp.copy())
        self.neuron_activate_dense=np.concatenate(self.neuron_activate_dense,axis=1)
        self.neuron_activate_gl=np.concatenate(self.neuron_activate_gl,axis=1)
        neuron_num= self.neuron_activate_dense.shape[-1] + self.neuron_activate_gl.shape[-1]
        upper=(self.neuron_activate_dense>self.t)
        upper_gl=(self.neuron_activate_gl>self.t)
        from torch_scatter import scatter
        import torch
        size = int(np.max(batch) + 1)
        upper_gl = scatter(torch.tensor(upper_gl), torch.tensor(batch), dim=0, dim_size=size, reduce='add').numpy()
        upper_gl = upper_gl > 0
        upper = np.concatenate(( upper, upper_gl), axis=1)
        score = np.sum(upper, axis=1)/neuron_num

        return np.argsort(np.sum(upper,axis=1))[::-1], score[ np.argsort(np.sum(upper,axis=1))[::-1] ]
