import numpy as np
import pandas as pd
from parameter import TuneParam,TrainParam

class FedHybrid:
    def __init__(self, func, fn_star):
        self.func = func
        self.A = self.func.A
        _, self.ndim = self.A[0].shape
        self.y = self.func.y
        self.nclient = len(self.y)
        self.fn_star = fn_star
    
    def tune_innerloop(self, param,innerIterations,func):
        print(str(func),"nclient",self.nclient)
        tune_fedh_innerloop = []
        a_range, b_range, a2_range, b2_range, mu_range, K= param.alpha1_range, param.beta1_range, param.alpha2_range, param.beta2_range, param.mu_range, param.K
        print('a_range',a_range,'b_range',b_range,'a2_range',a2_range,'b2_range',b2_range,'mu_range',mu_range,'newton_client',param.client_Newton)
        for u in mu_range:
            for a in a_range:
                for b in b_range:
                    for a2 in a2_range:
                        for b2 in b2_range:
                            param = TrainParam(alpha1 = a, beta1 = b, alpha2 = a2, beta2 = b2, mu = u, K = K, 
                                                            client_gradient= param.client_gradient, client_Newton = param.client_Newton, initial_x=param.initial_x)
                            fn_fedh, k_fedh, _ ,success= func(param,innerIterations)
                            #print('alpha=', a, 'beta=', b, 'alpha2=', a2, 'beta2=', b2, 'mu=', u, 'fn_fedh_last=', fn_fedh[-1], 'k_fedh=', k_fedh,'succuss',success)
                            if success==1 :
                                tune_fedh_innerloop.append([a, b, a2, b2, u, k_fedh, fn_fedh[-1]])
                                #plt.plot(np.log(fn_fedh))
                                #plt.savefig('Logistic_Synthetic/tune/FedH' + str(len(param.client_Newton)) + '/tune' + str(a) + '_' + str(b) + '_' + str(a2) + '_' + str(b2) + '_' + str(u) + '.pdf')
                                #plt.clf()
                                K=k_fedh
                                #break
                            
        
        return tune_fedh_innerloop
    def tune_innerloop2(self, param,innerIterations,func):#save multiple optimal stepsizes
        print(str(func),"nclient",self.nclient)
        tune_fedh_innerloop = []
        a_range, b_range, a2_range, b2_range, mu_range, K= param.alpha1_range, param.beta1_range, param.alpha2_range, param.beta2_range, param.mu_range, param.K
        print('a_range',a_range,'b_range',b_range,'a2_range',a2_range,'b2_range',b2_range,'mu_range',mu_range,'newton_client',param.client_Newton)
        for u in mu_range:
            for a in a_range:
                for b in b_range:
                    for a2 in a2_range:
                        for b2 in b2_range:
                            param = TrainParam(alpha1 = a, beta1 = b, alpha2 = a2, beta2 = b2, mu = u, K = K, 
                                                            client_gradient= param.client_gradient, client_Newton = param.client_Newton, initial_x=param.initial_x)
                            fn_fedh, k_fedh, _ ,success= func(param,innerIterations)
                            #print('alpha=', a, 'beta=', b, 'alpha2=', a2, 'beta2=', b2, 'mu=', u, 'fn_fedh_last=', fn_fedh[-1], 'k_fedh=', k_fedh,'succuss',success)
                            tune_fedh_innerloop.append([a, b, a2, b2, u, k_fedh, fn_fedh[-1],success])
        return tune_fedh_innerloop
        
    def train_innerloop(self, param,innerIterations):
        success=0;
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K
        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1]) #ndim rows, 1 column, nlient arrays
        new_x = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        for k in range(K):
            #print('iteration ',k)
            for j in range(innerIterations):
                for i in param.client_gradient: 
                    g = self.func.local_grad(x[i], i) # local gradient
                    new_x[i] = x[i] - alpha1 * (g - dual[i] + mu * (x[i] - x0))
                    dual[i] = dual[i] + beta1 * (x0 - x[i])
                    x[i] = new_x[i]
                for i in param.client_Newton: 
                    g = self.func.local_grad(x[i], i) # local gradient
                    H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                    new_x[i] = x[i] - alpha2 *(np.linalg.inv(H)) .dot(g - dual[i] + mu * (x[i] - x0))
                    dual[i] = dual[i] + beta2 * H.dot(x0 - x[i])
                    x[i] = new_x[i]
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
            if fn[-1] > 1e15:
                break

            if np.log(fn[-1] - self.fn_star) < -20:
                success =1;
                break
        return np.array(fn) - self.fn_star, k, x,success    #substraction elementwise

    def train_innerloop_gradient_only(self, param,innerIterations):
        success=0;
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K
        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1]) #ndim rows, 1 column, nlient arrays
        new_x = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        for k in range(K):
            #print('iteration ',k)
            for j in range(innerIterations):
                for i in param.client_gradient: 
                    g = self.func.local_grad(x[i], i) # local gradient
                    new_x[i] = x[i] - alpha1 * (g - dual[i] + mu * (x[i] - x0))
                    dual[i] = dual[i] + beta1 * (x0 - x[i])
                    x[i] = new_x[i]
            for i in param.client_Newton: 
                g = self.func.local_grad(x[i], i) # local gradient
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                #new_x[i] = x[i] - alpha2 *(np.linalg.inv(H)) .dot(g - dual[i] + mu * (x[i] - x0))
                new_x[i] = x[i] - alpha2 *np.linalg.solve(H,g - dual[i] + mu * (x[i] - x0))
                dual[i] = dual[i] + beta2 * H.dot(x0 - x[i])
                x[i] = new_x[i]
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
            if fn[-1] > 1e15:
                break
            if np.log(fn[-1] - self.fn_star) < -20:
                success =1;
                break
        return np.array(fn) - self.fn_star, k, x,success    #substraction elementwise
    
    
    ############################################################################
    def train_innerloop_gradient_only_quadratic(self, param,innerIterations):
        success=0;
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K
        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1]) #ndim rows, 1 column, nlient arrays
        new_x = np.zeros([self.nclient, self.ndim, 1])
        Hess=np.zeros([self.nclient,self.ndim,self.ndim])
        Hess_inv=np.zeros([self.nclient,self.ndim,self.ndim])
        self.func.initialize()
        for i in range(self.nclient):
            x[i] = x0
            Hess[i] =self.func.local_hess2(i) + mu * np.identity(self.ndim)
            #print(i ,' ', Hess[i])
            Hess_inv[i] = np.linalg.inv(Hess[i])
            
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        for k in range(K):
            #print('iteration ',k)
            for j in range(innerIterations):
                for i in param.client_gradient: 
                    g = self.func.local_grad2(x[i], i) # local gradient
                    new_x[i] = x[i] - alpha1 * (g - dual[i] + mu * (x[i] - x0))
                    dual[i] = dual[i] + beta1 * (x0 - x[i])
                    x[i] = new_x[i]
            for i in param.client_Newton: 
                g = self.func.local_grad2(x[i], i) # local gradient
                H = Hess[i]
                new_x[i] = x[i] - alpha2 *Hess_inv[i] .dot(g - dual[i] + mu * (x[i] - x0))
                #new_x[i] = x[i] - alpha2 *np.linalg.solve(H,g - dual[i] + mu * (x[i] - x0))
                dual[i] = dual[i] + beta2 * H.dot(x0 - x[i])
                x[i] = new_x[i]
                
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
            if fn[-1] > 1e15:
                break
            if np.log(fn[-1] - self.fn_star) < -20:
                success =1;
                break
        return np.array(fn) - self.fn_star, k, x,success    #substraction elementwise
    def train_innerloop_lambdaBefore(self, param,innerIterations):
        success=0;
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K
        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1]) #ndim rows, 1 c                                                  d'olumn, nlient arrays
        new_x = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        for k in range(K):
            #print('iteration ',k)
            for i in param.client_gradient:
                x_old=x[i].copy()
                for j in range(innerIterations):
                    g = self.func.local_grad(x[i], i) # local gradient
                    new_x[i] = x[i] - alpha1 * (g - dual[i] + mu * (x[i] - x0))
                    x[i] = new_x[i]
                dual[i] = dual[i] + beta1 * (x0 - x_old)
            for i in param.client_Newton: 
                g = self.func.local_grad(x[i], i) # local gradient
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                new_x[i] = x[i] - alpha2 *np.linalg.solve(H,g - dual[i] + mu * (x[i] - x0)) # can almost double the speed 
                #new_x[i] = x[i] - alpha2 *(np.linalg.inv(H)) .dot(g - dual[i] + mu * (x[i] - x0))
                dual[i] = dual[i] + beta2 * H.dot(x0 - x[i])
                x[i] = new_x[i]
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
            if fn[-1] > 1e15:
                break

            if np.log(fn[-1] - self.fn_star) < -20:
                success =1;
                break
        return np.array(fn) - self.fn_star, k, x,success    #substraction elementwise

    def train_innerloop_lambdaAfter(self, param,innerIterations):
        success=0;
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K
        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1]) #ndim rows, 1 column, nlient arrays
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        for k in range(K):
            #print('iteration ',k)
            for i in param.client_gradient:
                dual[i] = dual[i] + beta1 * (x0 - x[i])
                for j in range(innerIterations): 
                    g = self.func.local_grad(x[i], i) # local gradient
                    x[i] = x[i] - alpha1 * (g - dual[i] + mu * (x[i] - x0))
            for i in param.client_Newton: 
                g = self.func.local_grad(x[i], i) # local gradient
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                dual[i] = dual[i] + beta2 * H.dot(x0 - x[i])
                x[i] = x[i] - alpha2 *(np.linalg.inv(H)) .dot(g - dual[i] + mu * (x[i] - x0))
            
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
            if fn[-1] > 1e15:
                break

            if np.log(fn[-1] - self.fn_star) < -20:
                success =1;
                break
        return np.array(fn) - self.fn_star, k, x,success    #substraction elementwise

    def asynchronousTrain2 (self,param,poissonBeta):
        np.random.seed(2022)
        poissonLambda=[1/num for num in poissonBeta]
        sumLambda=sum(poissonLambda)
        poissonProb=[num/sumLambda for num in poissonLambda]
        success=0
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K
        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1])
        new_x = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        agentSequence=np.random.choice(self.nclient, int(K*self.nclient*(1/min(poissonProb))), p=poissonProb)
        #print('poissonProb',poissonProb)
        #print(agentSequence)
        for k in range(int(K*self.nclient*(1/min(poissonProb)))):
            
            i = agentSequence[k]
            if i in param.client_gradient: 
                
                g = self.func.local_grad(x[i], i) # local gradient
                new_x[i] = x[i] - (alpha1) * (g - dual[i] + mu * (x[i] - x0))
                dual[i] = dual[i] + (beta1) * (x0 - x[i])
                x[i] = new_x[i]
                
            if i in param.client_Newton:
                g = self.func.local_grad(x[i], i) # local gradient
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                new_x[i] = x[i] - (alpha2) *(np.linalg.inv(H)) .dot(g - dual[i] + mu * (x[i] - x0))
                dual[i] = dual[i] + (beta2) * H.dot(x0 - x[i])
                x[i] = new_x[i]
               
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            fn.append(self.func.global_func(x0))
    
            if fn[-1] > 1e15:
                break

            if np.log(fn[-1] - self.fn_star) < -20:
                success =1;
                break
    
        return np.array(fn) - self.fn_star, k, x,success    #substraction elementwise
'''  
    def train_innerloop_lambdaAfter(self, param,innerIterations):
        success=0;
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, 2**param.mu, param.K
        x0 = param.initial_x
        x = np.zeros([self.nclient, self.ndim, 1]) #ndim rows, 1 column, nlient arrays
        new_x = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            x[i] = x0
        dual = np.zeros([self.nclient, self.ndim, 1])
        fn = []
        for k in range(K):
            #print('iteration ',k)
            for i in param.client_gradient:
                for j in range(innerIterations): 
                    g = self.func.local_grad(x[i], i) # local gradient
                    x[i] = x[i] - alpha1 * (g - dual[i] + mu * (x[i] - x0))


            for i in param.client_Newton: 
                g = self.func.local_grad(x[i], i) # local gradient
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                x[i] = x[i] - alpha2 *(np.linalg.inv(H)) .dot(g - dual[i] + mu * (x[i] - x0))
            x0 = x.mean(axis = 0) - dual.mean(axis = 0)/mu
            for i in param.client_gradient:
                dual[i] = dual[i] + beta1 * (x0 - x[i])
            
            for i in param.client_Newton:
                H = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                dual[i] = dual[i] + beta2 * H.dot(x0 - x[i])

            fn.append(self.func.global_func(x0))
            if fn[-1] > 1e15:
                break

            if np.log(fn[-1] - self.fn_star) < -20:
                success =1;
                break
        return np.array(fn) - self.fn_star, k, x,success    #substraction elementwise
''' 
    