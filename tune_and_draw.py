import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from parameter import TuneParam,TrainParam
class TuneAndDraw:
    def __init__(self, nclient, initial_x,setup):
        self.initial_x = initial_x
        self.nclient=nclient
        self.setup = setup
        
    def readPa(self,ns,innerloop):
        file=pd.read_csv('readfile.csv',sep = '|')
        drawFile = (file.loc[file["setup"]==self.setup])["drawFile"].to_numpy()[0]
        #print('drawFile',drawFile)
        data = pd.read_csv(drawFile)
        print("data",data.head()["nodes"].dtypes)
        
        df = pd.DataFrame()  
        for nsecond in ns:
            row = data.loc[(data["NewtonAgentnum"]==nsecond) & (data["innerloop"]==innerloop) & (data["nodes"]==self.nclient)]
            #print(row)
            if row.empty:
                print('sorry,'+str(nsecond)+ ' of agents doing Newton''s method or '+ str(innerloop) +' of innerloops does not exist or ' +str(self.nclient)+" of nodes does not exist" )
            else:
                df = pd.concat([df, row])
        return (df)

    def readAsyncPa(self,ns,betaType):
        file=pd.read_csv('readfile.csv',sep = '|')
        drawFile = (file.loc[file["setup"]==self.setup])["drawFile"].to_numpy()[0]
        #print('drawFile',drawFile)
        data = pd.read_csv(drawFile)
        print("data",data.head()["nodes"].dtypes)
        df = pd.DataFrame()  
        for nsecond in ns:
            row = data.loc[(data["NewtonAgentnum"]==nsecond) & (data["betaType"]==betaType) & (data["nodes"]==self.nclient)]
            print(row)
            if row.empty:
                print('sorry,'+str(nsecond)+ ' of agents doing Newton''s method or '+ str(betaType) +' of betaType does not exist or ' +str(self.nclient)+" of nodes does not exist" )
            else:
                df = pd.concat([df, row])
        return (df)

    def saveWithMoreRun(self,df,K,func):
        MoreRun=pd.DataFrame()
        for i in range(len(df)):
            row = df.iloc[[i]].copy()
            row.loc[:,'K']=K
            param,K,nsecond,innerIterations = self.train_parameter(row)
            fn_list, k_iter, _,success = func(param,innerIterations)
            row.loc[:,'lastIterationValue'] = fn_list[-1]
            row.loc[:,'K'] = k_iter
            row.loc[:,'flag'] = success
            MoreRun=pd.concat([MoreRun,row])
        return MoreRun

    def readTunePa(self,nsecond,large_condition_num = 0):
        if large_condition_num == 0:        
            file=pd.read_csv('readfile.csv',sep = '|')
            tuneFile = (file.loc[file["setup"]==self.setup])["tuneFile"].to_numpy()[0]
            print(self.nclient)
            data = pd.read_csv(tuneFile)   
            row = data.loc[(data["NewtonAgentnum"]==nsecond) & (data["nodes"]==self.nclient)]
        else:
            data =pd.read_csv('./setup_1_result/train/tune_setup1_heter2.csv')
            row = data.loc[(data["NewtonAgentnum"]==nsecond) & (data["nodes"]==self.nclient) & (data["large_condition_num"]==large_condition_num)]
        if row.empty:
            print('sorry,'+str(nsecond)+ ' of agents doing Newton''s method does not exist')
            return (0,[],[0],[0],[0],[0],[0])
        client_Newton = list(map(int, row['newtonAgent'].to_numpy()[0].strip('[]').split()))
        alpha1_range = list(map(int, row['a1_range'].to_numpy()[0].strip('[]').split()))
        beta1_range = list(map(int, row['b1_range'].to_numpy()[0].strip('[]').split()))
        alpha2_range = list(map(int, row['a2_range'].to_numpy()[0].strip('[]').split()))
        beta2_range = list(map(int, row['b2_range'].to_numpy()[0].strip('[]').split()))
        mu_range=list(map(int, row['mu_range'].to_numpy()[0].strip('[]').split()))
        return(row["K"].to_numpy()[0].item() ,np.array(client_Newton),alpha1_range,beta1_range,alpha2_range,beta2_range,mu_range)
    
    def tuneInitialize(self,nsecond,large_condition_num = 0):
        K,client_Newton,alpha1_range,beta1_range,alpha2_range,beta2_range,mu_range= self.readTunePa(nsecond,large_condition_num)
        client_gradient = np.setdiff1d(range(self.nclient), client_Newton) # clients that perform gradient updates
        #client_Newton =  np.setdiff1d(range(self.nclient), client_gradient)  # redundent
        tunePa=TuneParam(alpha1_range=alpha1_range,
                        beta1_range=beta1_range,
                        alpha2_range=alpha2_range,
                        beta2_range=beta2_range,
                        mu_range=mu_range,
                        K=K,
                        client_gradient=client_gradient,
                        client_Newton=client_Newton,
                        initial_x=self.initial_x)   
        return tunePa
    
    def train_parameter(self,row):
        nsecond,alpha1,beta1,alpha2,beta2,mu,K,client_Newton,innerIterations=  self.rowSplitTrain(row)
        client_gradient = np.setdiff1d(range(self.nclient), client_Newton)
        param = TrainParam(alpha1=alpha1,
                  beta1=beta1,
                  alpha2=alpha2,
                  beta2=beta2,
                  mu=mu,
                  K=K,
                  client_gradient=client_gradient,
                  client_Newton=client_Newton,
                  initial_x=self.initial_x)
        return param,K,nsecond,innerIterations
        
    def draw(self,df,func,title,filename,save,asyn,possionBeta):
        fig=[]
        existedNsecond = []
        for i in range(len(df)):
            row = df.iloc[[i]]
            param,K,nsecond,innerIterations = self.train_parameter(row)
            if K: 
                existedNsecond.append(nsecond)
                if asyn ==0:
                    fn_list, k_iter, _,success = func(param,innerIterations)
                else:
                    fn_list, k_iter, _,success = func(param,possionBeta)
                print(fn_list[-1],k_iter,success)
                if (success !=1):
                    print(nsecond,'not succussful')
                fig.append(fn_list)
        for i in range (len(existedNsecond)):
            plt.plot(np.log(fig[i]),label='FedH- %s' % existedNsecond[i])
        plt.legend()
        plt.title(title)
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.xlabel('Number of communications, '  + r'$r$')
        plt.ylabel('Log Optimality Gap, ' + r'$\log(f(x_0^r) - f^*)$')  
        if save ==0:
            plt.show()
        else:
            plt.savefig(filename)

    def tune(self,method,ns,inner,func,name,large_condition_num=0):
        BestPa=pd.DataFrame()
        for innerIterations in inner:
            for nsecond in ns:
                print('large_condition_num',large_condition_num)
                param = self.tuneInitialize(nsecond,large_condition_num)
                print(self.nclient,' ',nsecond,': ',param.alpha1_range,param.beta1_range,param.alpha2_range,param.beta2_range,param.mu_range,param.K,param.client_Newton)
                print('client_gradient',param.client_gradient)
                print('client_newton',param.client_Newton)
                fn_list= method.tune_innerloop(param,innerIterations,func)
                df = pd.DataFrame(fn_list)
                if(df.empty):
                    continue
                newRow = df.iloc[[len(df)-1]]
                newRow=newRow.assign(newtonAgent=[str(param.client_Newton)])
                newRow=newRow.assign(innerloop=[str(innerIterations)])
                newRow = newRow.assign(method = [name])
                newRow = newRow.assign(setup = [str(self.setup)])
                newRow = newRow.assign(nodes = str(self.nclient))
                newRow.insert(0, 'NewtonAgentnum', nsecond)
                BestPa= pd.concat([BestPa,newRow], axis = 0,
                   ignore_index=True)
        if (not BestPa.empty):
            BestPa.columns = ["NewtonAgentnum","a1","b1","a2","b2","mu","K","lastIterationValue","newtonAgent","innerloop","method","setup","nodes"]
        return(BestPa)
    
    def tune_save_all(self,method,ns,inner,func,name,large_condition_num=0):
        BestPa=pd.DataFrame()
        for innerIterations in inner:
            for nsecond in ns:
                param = self.tuneInitialize(nsecond,large_condition_num)
                print(self.nclient,' ',nsecond,': ',param.alpha1_range,param.beta1_range,param.alpha2_range,param.beta2_range,param.mu_range,param.K,param.client_Newton)
                print('client_gradient',param.client_gradient)
                print('client_newton',param.client_Newton)
                fn_list= method.tune_innerloop2(param,innerIterations,func)
                df = pd.DataFrame(fn_list)
                df.columns=["a1","b1","a2","b2","mu","K","lastIterationValue","flag"]
                if(df.empty):
                    continue
                newRow = df
                newRow=newRow.assign(newtonAgent=str(param.client_Newton))
                newRow=newRow.assign(innerloop=str(innerIterations))
                newRow = newRow.assign(method = name)
                newRow = newRow.assign(setup = str(self.setup))
                newRow = newRow.assign(nodes = str(self.nclient))
                newRow.insert(0, 'NewtonAgentnum', nsecond)
                BestPa= pd.concat([BestPa,newRow], axis = 0,
                   ignore_index=True)
        if (not BestPa.empty):
            BestPa.columns = ["NewtonAgentnum","a1","b1","a2","b2","mu","K","lastIterationValue","flag","newtonAgent","innerloop","method","setup","nodes"]
        return(BestPa)
    
    def rowSplitTrain(self,row):
        client_Newton = list(map(int, row['newtonAgent'].to_numpy()[0].strip('[]').split()))
        return(row["NewtonAgentnum"].to_numpy()[0].item()\
            ,row["a1"].to_numpy()[0].item(), row["b1"].to_numpy()[0].item(), \
           row["a2"].to_numpy()[0].item(), row["b2"].to_numpy()[0].item(), \
           row["mu"].to_numpy()[0].item(),row['K'].to_numpy()[0].item()+1,np.array(client_Newton),\
           row["innerloop"].to_numpy()[0].item())
        
    def direct_train(self,alpha1,beta1,alpha2,beta2,K,mu,client_Newton):
        client_gradient = np.setdiff1d(range(self.nclient), client_Newton)
        param = TrainParam(alpha1=alpha1,
                  beta1=beta1,
                  alpha2=alpha2,
                  beta2=beta2,
                  mu=mu,
                  K=K,
                  client_gradient=client_gradient,
                  client_Newton=client_Newton,
                  initial_x=self.initial_x)
        return param
