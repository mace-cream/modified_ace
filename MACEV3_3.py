# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:46:32 2017

@author: Jialin Li
"""

import numpy as np
import pandas as pd
import sys

'''
This is the version 3 of MACE, here are the tips:
    

1. The input training data should be a 2-D numpy.ndarray, with a shape of 

   (nsample, kfields). Each row is a training sample, each colomun is a field.
   
   Any data type is ok only if it can be contained in numpy.ndarray.


2. The input schedule decides the update rule. Default is full mace.
 
   For eaxample, suppose that the shape of input training data is (1000, 4), 
   
   then the default initialized schedule whould be:
       
                    [ 1 , [2, 3, 4] ]
                    [ 2 , [1, 3, 4] ]
                    [ 3 , [1, 2, 4] ]
                    [ 4 , [1, 2, 3] ]     
                    
   for the case that there are two group of data fields, if the input 
   
   schedule is [[1,2], [3,4]] , then the initailized schedule would be :

                    [ 1 , [3, 4] ]
                    [ 2 , [3, 4] ]
                    [ 3 , [1, 2] ]
                    [ 4 , [1, 2] ]
                    
   notice: non-default schedule could be run through successfully, but results
           
           are not right till now, since the inapplicability of regularization
           
           and orthogonation.                     


3. Test data's feature returned by the function getFeature will be a 
   
   3-D numpy.ndarray, with a shape of (nsample, kfield, fDim), fDim is the 
   
   dimensionality of features.
 

4. compute the singular values the same way as MACEV2, however seems different 
   
   from what written in the paper? 
   
5. weights and other attributes waited to be implemented in future version.


'''

class MACE:
    
    '''
    Attributes
    
       X     = input data, 2-D numpy.ndarray with a shape of (nsample, kfields)
               each row is a training sample, each colomun is a field
               
       P     = empirical distribution of X, computed but not used now
 
       K     = dimensionality of features

    schedule = update schedule
    
    weights  = copy from the input, to be used in weighted updates
               dont know how to use now
                
                    
       f     = list of kfields DataFrames, each a function, always normalized
    
               for example, X0 is a radom variable, X0 = a,b,c,d
              
                             index  |  value
                             ---------------
               then    f[0] =  X0   |  f(X0)
                             ---------------     
                               a    |  f(a)
                             ---------------                                
                               b    |  f(b)
                             ---------------         
                               c    |  f(c)
                             ---------------
                               d    |  f(d)
                             ---------------                            
    
   sig_values = store singular values  
   
   covariance = store convariance for every dimension
    '''  
    
    def __init__ (self,X=np.array([]), K=5, schedule=[], weights=np.array([])):
        
        self.X = X
        
        self.P = self.getEmpiricals(X)
        
        self.K= K
        
        self.schedule = self.init_schedule(schedule)
        
        if weights.size==0:
            self.weights = np.ones(X.shape)
        elif weights.size == X.shape[0]:
            self.weights=weights
        else:
            sys.exit('weights error')
        
        self.f = []
        
        self.sig_values = []
        
        self.covariance = []
        
        
    def init_schedule(self,sch):
        
        # schedule for default 
        if sch == []:
            sch = list(range(self.X.shape[1]))
            schedule = [sch,sch]

        # schedule for the case of two grouped mace
        elif len(sch)==2:
            schedule = sch
        
        # shedule for other situations of mace, waitted to be implemented
        else:
            sys.exit('only support full mace and two grouped mace now')

        print('schedule initialized')
        return schedule        
   

    def regulate_feature_function(self,fx):
       
        #zero-means
        fx = fx - np.mean(fx,axis=0, keepdims=True)
        
        # union-variance
        fx_var = np.mean(np.sum(np.square(fx),axis=1, keepdims=True),\
                         axis = 0,keepdims=True)
        fx = fx/np.sqrt(fx_var)
    
        return fx 

    
    def schimidt_process(self,fx):
        
        ## gram_schimidt of features in different dimensions 
        for i in range(fx.shape[2]-1):

            alpha = fx[:,:,[i+1]]
            
            beta = fx[:,:,:i+1]
            
            k= np.sum(np.mean(np.multiply(alpha,beta), axis=0,\
                                        keepdims = True),axis=1,keepdims=True)
            
            fx[:,:,[i+1]] = alpha - np.sum(k*beta, axis=2, keepdims=True)
            
        return fx   
        
  
    
    def run(self, nIter=5):
        
        # initialization of data's features
        fx_shape = list(self.X.shape)
        fx_shape.append(self.K)
        fx = np.random.normal(0,1,fx_shape)       
        new_fx = np.zeros(fx.shape) 
        
        ## do ACE
        #1. select data for conditional expectation 
        #2. do conditiobnal expectation
        #3. collect the new feature 
        
        print('--------mace begin---------')
        for it in range(nIter):           
           
            print('---------Iter ',it,' --------')    

            for s in range(len(self.schedule)):
                
                # some prepare for select data
                schedule = self.schedule[s]
                source = self.schedule[1-s]
                fx_sum = np.sum(fx[:,source,:],axis=1)
                
                for i in schedule:
                    #1. select data
                    if schedule == source:
                        fx_select = fx_sum - fx[:,i,:]
                    else :
                        fx_select = fx_sum.copy()
                    
                    #2. do conditiobnal expectation
                    fx_select = pd.DataFrame(fx_select)
                    x_index = self.X[:,i].copy()
                    fx_new = fx_select.groupby(x_index).mean()
    
                    #3. collect the new feature 
                    new_fx[:,i,:] = fx_new.loc[x_index].values
            
            # use the new fx to take palce of the old one
            fx = new_fx.copy()
                   
            ## zero-mean and union-variance
            fx = self.regulate_feature_function(fx)
                    
            ## gram_schimidt
            fx = self.schimidt_process(fx)  

        print('get feature function')
        #conclude funtions from features
        self.f = self.get_feature_function(fx)
        #compute covariance matrix for every dimension
        self.covariance = self.getCovariance(fx)
        #compute sigular values
        self.sig_values = self.getSig()

        print('mace finished')
 
    

    def get_feature_function(self,fx):
        
        '''
         all previous running procedure are based on a huge pandas dataframe
         with a shape  of (nsample, kfield, fDim)
         
         this function is used to conclude feature functions from features
        '''
        feature_function = []
       
        for i in range(fx.shape[1]):
         
            f = pd.DataFrame(fx[:,i,:], index = self.X[:,i])
            
            # get the function
            f.drop_duplicates(inplace=True)
            
            f.sort_index(inplace=True)
            
            feature_function.append(f)
              
        return feature_function 


    def getCovariance(self,fx):
        
        #initialize covariance
        nsample,kfields,fdim = fx.shape
        covariance = np.zeros([kfields,kfields,fdim])
        
        #get covariance matrix for every dimension
        for i in range(fx.shape[2]):
            fx_i  =fx[:,:,i]
            covariance[:,:,i] = np.cov(fx_i.T)
            
        return covariance
    
 
    def getSig(self):
        
        '''
        compute sigular values as MACEV2 did, which seems not exactlly the same 
        as it is in the paper
        '''
        covariance = self.covariance
        
        sig_values = []
        
        for i in range(covariance.shape[2]):
            cov = covariance[:,:,i]
            sig = np.sqrt((cov.sum()+1)/2.0)
            sig_values.append(sig)
        
        sig_values = np.array(sig_values)
        
        return sig_values    


    def getEmpiricals(self,X):
       
        # get empirical distributions of training data
        P=[]
        for i in range(X.shape[1]):        
            x = pd.Series(X[:,i])
            counts = x.groupby(x.values).count()
            prob = counts.to_frame()/counts.sum()
            P.append(prob)
        return P
               

    def getFeature(self,X,fields = []):
        
        '''
        
        test data must be a 2-D numpy.ndarray with the same number of field as
        
        training data.feature returned is a 3-D numpy.ndarray, with a shape of 
        
        (nsample, kfield, fDim), fDim is the dimensionality of features.
        
        you can choose which random variables' features to return.
        
        
        For example, you do mace of X1,X2,X3, and only want features of X2,X3,
        
        then pass in fields = [2,3]. the number stands for the positon of 
        
        columns of these fields in test data.
        
        '''
        nsample,nfields = X.shape
        
        # default is taking out all features 
        if fields == []:
           xfields = range(nfields) 
        else :
            #otherwise take out wanted features
            xfields = fields
        feature = np.zeros([nsample,len(xfields),self.K])
        
        pos = 0
        for i in xfields:
            fx = self.f[i].loc[X[:,pos]]
            feature[:,pos,:] = fx.values
            pos+=1
            
        return feature



if __name__ == "__main__":
    
    X=np.random.randint(0, 6, 100)
    Y=np.random.randint(0, 6, 100)
    Z=np.random.randint(0, 6, 100)
    
    Bii = np.identity(6)
    
    BXY = np.zeros((6,6))
    for i in range(100):
        BXY[X[i], Y[i]]+=1
        
    BXY /= 100
    
    px  = np.sum(BXY ,axis=1, keepdims=True)
    py  = np.sum(BXY ,axis=0, keepdims=True)
    
    BXY = BXY / np.sqrt(px) / np.sqrt(py)
    
    BYZ = np.zeros((6,6))
    for i in range(100):
        BYZ[Y[i], Z[i]]+=1
    
    BYZ /= 100
    
    py  = np.sum(BYZ ,axis=1, keepdims=True)
    pz  = np.sum(BYZ ,axis=0, keepdims=True)
    
    BYZ = BYZ / np.sqrt(py) / np.sqrt(pz)

    BXZ = np.zeros((6,6))
    for i in range(100):
        BXZ[X[i], Z[i]]+=1

    BXZ /= 100
    px  = np.sum(BXZ ,axis=1, keepdims=True)
    pz  = np.sum(BXZ ,axis=0, keepdims=True)
    
    BXZ = BXZ / np.sqrt(px) / np.sqrt(pz)
    
    B01 = np.concatenate([Bii, BXY, BXZ], axis = 1)
    B02 = np.concatenate([BXY.T, Bii, BYZ], axis = 1)
    B03 = np.concatenate([BXZ.T, BYZ.T, Bii], axis = 1)
    B = np.concatenate([B01, B02, B03], axis=0)
        
    eig, vec = np.linalg.eig(B)
    fx = vec[:6,:] / np.sqrt(px)
    
    XYZ = np.array([X,Y,Z])
    XYZ = XYZ.T
    
    model = MACE(XYZ,K=15)
    model.run(200)
    fx1 = model.f[0].values
    
    covariance = model.covariance
    sig = model.sig_values
    P = model.P
  
    test = model.getFeature(XYZ,[1,2])