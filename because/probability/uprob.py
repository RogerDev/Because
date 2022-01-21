""""UPROB is an universal probability module that encompasses all existing methods, JPROB and FPROB. The main funcitonality is to have a 'lambda'
    parameter such that out of N variables/dimensions, lambda% of variables are filtered and JPROB is applied to the rest. Important comparisions:
    UPROB(0) == JPROB
    UPROB(99) == FPROB1 (filter all but 1 conditionals and return 2D RKHS result)
    UPROB(100) == FPROB2 (filter all variables and return unidimensional RKHS result)
"""


from because.probability.pdf import PDF
from math import ceil, floor

tau = 50  # The threshold for Ncond used for automatic Lambda selection
          # Higher tau implies more kernel.  Lower -> more filtering
mintau = 5

# Calculates and returns (lambda, Dfilter, Ntarg)
def calcParms(lambdaIn, N, Dquery):
    # Automatic selction of Lambda. When the datapoints are abundant, lambda=100 (DPROB) 
    # is utilized, else lambda is reduced to predominantly use JPROB (lambda=25 for example)
    # Only if lmbda was passed as None
    Dcond = Dquery - 1
    Ncond = N**(1/Dcond)
    if lambdaIn is None:
        # Automatically determine
        if Ncond < tau:
            if Ncond < mintau:
                lmbda = 0
            else:
                lmbda = 100 * (Ncond / tau)
        else:
            lmbda = 100
    else:
        # Use the provided Lambda
        lmbda = lambdaIn
    Dfilter = int(Dcond * lmbda * 0.01)
    Ntarg = N**((Dquery-Dfilter)/Dquery) / (1.5**(Dcond-Dfilter))


    return (lmbda, Dfilter, Ntarg)




class UPROB:
    def __init__(self, ps, rvName, condSpecs, delta = 3, s=1.0,lmbda=None,rangeFactor = None):
        #print('UProb: condSpecs = ', condSpecs, ', rvName = ', rvName)
        self.ps = ps                        # The ProbSpace that called us
        self.data = ps.ds                   # data in the form of a dictionary
        self.delta = delta                  # deviation of test points
        self.rvName = rvName                # target variable
        self.condSpecs = condSpecs          # The conditionals
        self.s = s                          # smoothness factor
        self.lmbda = lmbda                  # % of variables to be filtered
        self.R1 = None                      # cached rkhsMV class
        self.R2 = None                      # cached rkhsMV class2
        self.minPoints = None               # min and max points for filteration
        self.maxPoints = None
        self.N = self.ps.N
        if rangeFactor is None:
            self.rangeFactor = .9
        else:
            self.rangeFactor = rangeFactor
        self.Dquery = len(condSpecs) + 1
        # Automatic selction of Lambda. When the datapoints are abundant, lambda=100 (DPROB) 
        # is utilized, else lambda is reduced to predominantly use JPROB (lambda=25 for example)
        # Only if lmbda was passed as None
        self.lmbda, self.Dfilter, self.Ntarg = calcParms(lmbda, self.N, self.Dquery)

    def distr(self):
        # We'll try to hit these filter levels, but if we don't get enough points,
        # we'll reduce our filtering.
        for Dfilter2 in range(self.Dfilter,-1,-1):
            #print("runing Dfilter2=",Dfilter2)
            if Dfilter2 > 0:
                filters = self.condSpecs[-Dfilter2:]
                if Dfilter2 == self.Dfilter:
                    Ntarg = self.Ntarg
                else:
                    Ntarg = self.N**((self.Dquery-Dfilter2)/self.Dquery)
                minPoints = floor(Ntarg * self.rangeFactor)
                maxPoints = ceil(Ntarg / self.rangeFactor)
                #print('filters, minPoints, maxPoints = ', filters, minPoints, maxPoints)
                #if Ndim < tau:           
                ss = self.ps.SubSpace(filters, minPoints=minPoints, maxPoints=maxPoints, power=self.ps.power,
                                                    density=self.ps.density, discSpecs=self.ps.discSpecs)
                #print('back from subspace')
                filteredData = ss.ds
                filteredLen = len(filteredData[self.rvName])
                #print('filteredData len = ', filteredLen)
                #if filteredLen < minPoints:
                if filteredLen < mintau:
                    print("not enough filter points for filterlen =",Dfilter2,filteredLen,minPoints,maxPoints)
                    # Not enough points returned from filter. Reduce Lambda and continue
                    continue
                rkhsFilters = self.condSpecs[:-Dfilter2]
                #print('rkhsFilters1 = ', rkhsFilters)
                #print('Dfilter = ', Dfilter2)
                break
            else:
                filteredData = self.data
                rkhsFilters = self.condSpecs
                ss = self.ps
                #print('Dfilter = 0')
                break
        # We want to return a PDF.  If we've filtered all, then return a discretized univariate pdf.
        # Otherwise, return a filtered multivariate.
        #print('rkhsFilters = ', rkhsFilters)
        if len(rkhsFilters) == 0:
            #print('No RKHS')
            # We don't need to use RKHS.  We have the final pdf
            # All that's left is the target variable
            dist = ss.distr(self.rvName)
        else:
            #print('RKHS: filteredData = ', len(filteredData[self.rvName]))
            # Return  a filtered multivariate pdf.
            #outPdf = pdf(numSamples, binList=None, isDiscrete = False, data=None, mvData = None, conds = [], filters = []
            # Arbitrary N
            N = 1000000
            #print('calling pdf')
            dist = PDF(N, binList = None, isDiscrete = False, data = None, mvData = filteredData, filters = rkhsFilters, rvName = self.rvName)
            #print('back from pdf')
            #print('Uprob: pdf mean = ', dist.E())
        return dist



    # def condE(self,target, Vals, lmbda = None):
    #     #Vals is a list of (x1,x2....xn) such that E(Y|X1=x1,X2=x2.....), same UI as rkhsmv
    #     if(lmbda == None):
    #         lmbda = self.lmbda 
    #     filter_len = floor((len(self.includeVars)-1)*lmbda*0.01)
    #     #print("filter len",filter_len)
    #     dims = len(Vals) + 1
    #     if(self.rangeFactor == None):
    #         self.rangeFactor = 0.8
    #     minminpoints = 5
        
        
    #     if(filter_len !=0):
    #         filter_vars = self.includeVars[-filter_len:]
    #         filter_vals = Vals[-filter_len:]
    #         include_vars = self.includeVars[1:-filter_len]
    #         self.minPoints = self.N**((dims-filter_len)/dims)*self.rangeFactor
    #         self.maxPoints = self.N**((dims-filter_len)/dims)/self.rangeFactor
    #         #print("minpoints,maxpoints=",self.minPoints,self.maxPoints)

    #     else:
    #         filter_vars = []
    #         filter_vals = []       
    #         include_vars = self.includeVars
        
    #     #print("filter vars:",filter_vars)
    #     #print("include vars:",self.includeVars[:-filter_len])
    #     #print("self:",self.R2.varNames,"cond",self.includeVars[:-filter_len])
                
        
    #     if(filter_len == (len(self.includeVars)-1) ):
    #         P = self.ps
    #         filter_vars = self.includeVars[1:]
    #         filter_vals = Vals            
    #         filter_data = []
    #         for i in range(filter_len):
    #             x = (filter_vars[i],filter_vals[i])
    #             filter_data.append(x)                    
    #         #print("minpoints,maxpoints:",self.minPoints,self.maxPoints)
    #         FilterData, parentProb, finalQuery = P.filter(filter_data,self.minPoints,self.maxPoints)
    #         X = FilterData[self.includeVars[0]]
    #         if(len(X)<self.minPoints):
    #             newlmbda = ceil((((lmbda*(dims-1)*0.01)-1)/(dims-1))*100) # update lambda =100 to lambda = 80
    #             print("not enough datapoints, newlmbda=",newlmbda)
    #             return self.condE(target, Vals, newlmbda)
    #         #print(len(X))
    #         if(len(X)!=0):
    #             return sum(X)/len(X)
    #         else:
    #             return 0
                
        
    #     elif(filter_len != 0 and self.r2IncludeVars != filter_vals):
    #         P = self.ps
    #         filter_data = []
    #         for i in range(filter_len):
    #             x = (filter_vars[i],filter_vals[i])
    #             #print(x)
    #             filter_data.append(x)                    
    #         FilterData, parentProb, finalQuery = P.filter(filter_data,self.minPoints,self.maxPoints)
    #         print("filter len",filter_len)
    #         print("filtered datapoints:",len(FilterData['B']))
    #         print("include vars:",self.includeVars[:-filter_len])
    #         X = FilterData[self.includeVars[0]]
    #         if(len(X)<self.minPoints or len(X)<=minminpoints):
    #             newlmbda = ceil((((lmbda*(dims-1)*0.01)-1)/(dims-1)) * 100)
    #             print("not enough datapoints, newlmbda=",newlmbda)
    #             return self.condE(target, Vals, newlmbda)
    #         self.R2 = RKHS(FilterData,includeVars=self.includeVars[1:-filter_len],delta=self.delta,s=self.s)
    #         self.r2filters = filter_vals          
        
    #     elif(self.R2==None):
    #         self.R2 = RKHS(self.data,includeVars=self.includeVars[1:],delta=self.delta,s=self.s)

    #     elif(self.R2.varNames != include_vars):
    #         self.R2 = RKHS(self.data,includeVars=self.includeVars[1:],delta=self.delta,s=self.s)
        
    #     if(filter_len !=0):
    #         return self.R2.condE(target,Vals[:-filter_len])
    #     else:
    #         return self.R2.condE(target, Vals)