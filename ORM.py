import numpy as np


class OrdinalRegressionMachine : 
    def __init__(self,f,X, y, ratio = 0.8):
        self.x = X
        self.y = y
        self.unique_classes = np.unique(self.y)
        self.num_classes = len(self.unique_classes)
        self.train_set = self.define_train_set()
        self.f = f # margin based classfier
        self.ratio = ratio
        
    def define_train_set(self):
        '''
        this function creates the appropriate training data indexes for the algorithm
        '''

        # let's define the number of combinations of labels (M * (M-1)) / 2
        P = [] # training set
        for i in range(self.num_classes): 
            for j in range(i+1, self.num_classes): 
                #comb_cls = self.unique_classes[comb]
                s_idx = np.where(self.y == i)[0]
                t_idx = np.where(self.y == j)[0]
                m_idx = np.where((self.y != i) & (self.y != j))[0] # rest of classes
                P.append([s_idx,m_idx,t_idx])
        return P
    
    def train_test_split(self, x, y):    
        '''
        function to split our data for training and testing
        ratio : % of data that goes to the training set
        '''
        n = len(x)
        n_samples = int(np.around(n*self.ratio, 0))
        n_data = np.arange(n)
        np.random.shuffle(n_data)
        idx_train = np.random.choice(n_data, n_samples, replace=False)
        idx_test = list(set(n_data) - set(idx_train))    
        X_train , X_test , Y_train, Y_test = x[idx_train, ], x[idx_test, ], y[idx_train, ], y[idx_test, ]
        return X_train , X_test , Y_train, Y_test

    def clean_classes(self):
        '''
        apply property 8.3.5 on the newly constructed classes, meaning 
        '''
        training_data = []
        for item in self.train_set: 
            s, m, t = item[0], item[1], item[2]
            tmp_x = np.copy(self.x)
            tmp_y = np.copy(self.y)
            tmp_y[s], tmp_y[m], tmp_y[t] = 1, 2, 3
            data = [tmp_x, tmp_y]
            training_data.append(data)
        
        return training_data
    

    def train(self):
        train_data = self.clean_classes()
        m = 1
        weights = []
        for data in train_data :
            X_train , X_test , Y_train, Y_test = self.train_test_split(data[0], data[1])
            w = self.f(X_train, Y_train)
            weights.append(w)
            m += 1 

            for inst in X_test : 
                tmp_x = [1, inst[0], inst[1]]
                res = np.dot(tmp_x , w)

        



        return



    def decision_function(self, prediction):
        '''
        hmm.. still unsure about this
        '''
        out = np.argmax(prediction)
        return out



    
    


        
        








