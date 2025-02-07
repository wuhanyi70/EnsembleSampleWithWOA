import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Define the objective function for optimization
from sklearn.model_selection import GridSearchCV



class Trainer():
    def __init__(self, grid, outputFile, min_class_filter_num, kernel, C, gamma):
        self.outputFileStream = open(outputFile, 'w')
        self.all_data = None
        self.train_data = None
        self.train_label = None
        self.train_id = None
        self.test_data = None
        self.test_label = None
        self.validate_data = None
        self.validate_label = None
        self.min_class_filter_num = 10
        self.classes = -1
        self.class_label = []
        self.class_amount = dict()
        self.min_class_amount = min_class_filter_num
        self.max_class_amount = -1
        self.imbalance_ratio = 1
        self.grid = grid
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        print("Output to", outputFile)

    def loadData(self, inputData):
        print("Start loading data from", inputData)
        self.outputFileStream.write("Data from "+inputData+"\n")

        # if data is .data format:
        data = None
        with open(inputData, 'r') as dataFile:
            data = pd.read_csv(dataFile, sep=',', header=None)
            #self.outputFileStream.write(data.head())

        # if data is excel format:
        # with open(inputData, 'r') as dataFile:
        #     data = pd.read_excel(dataFile, header=None)

        data.dropna(inplace=True)
        self.all_data = data

    print("Load complete")

    def prepareData(self):
        print("Start preparing train data")

        df = self.all_data
        l = df[df.shape[1] - 1].value_counts(ascending=True)
        i = 0
        j = 0
        flag = True
        for value in l:
            if value >= self.min_class_filter_num:
                if flag:
                    self.min_class_amount = value
                    flag = False
                self.class_amount[l.keys()[j]] = value
            else:
                i += 1
            j += 1
        print(self.class_amount)
        self.max_class_amount = l[l.keys()[-1]]
        self.class_label = l.keys()[i:].tolist()
        self.class_label.sort()
        self.classes = len(self.class_label)
        self.imbalance_ratio = float(l[l.keys()[-1]]) / float(self.min_class_amount)
        df = df[~df[df.shape[1] - 1].isin(l.keys()[0:i])]
        df = df.sample(frac=1)
        df = df.reset_index(drop=True)

        _df = df.sample(frac=0.9)
        df1 = df[~df.index.isin(_df.index)]
        df2 = _df.sample(frac=1.0/9.0)
        df3 = _df[~_df.index.isin(df2.index)]


        self.test_data = df1.iloc[:,:-1].to_numpy()
        self.test_label = df1.iloc[:,-1].to_numpy()
        self.validate_data = df2.iloc[:,:-1].to_numpy()
        self.validate_label = df2.iloc[:,-1].to_numpy()
        self.train_data = df3.iloc[:,:-1].to_numpy()
        self.train_label = df3.iloc[:,-1].to_numpy()
        self.train_id=list(range(len(self.train_data)))
        #self.train_data = _df.iloc[:,:-1].to_numpy()
        #self.train_label = _df.iloc[:,-1].to_numpy()

        print("Data preparation complete")


def change_back(listneedtochange, indexes):
    """apply indexes the list of index and create a sample vectors list to let the numbers in these location
    become 1 """
    l = len(listneedtochange)

    int_indexes = float_to_int_no_repeat(indexes,l)
    # print(int_indexes)
    for index in int_indexes:
        if abs(index) < l:
            listneedtochange[index] = 1
        else:
            xi =random.choice(np.where(listneedtochange == 0)[0])
            listneedtochange[xi] = 1
    return listneedtochange

# def revise_list_to_same_len(lista,listb,elementadd):
#     if len(lista) > len(listb):
#         x = random.choice(elementadd)
#         while x not in listb:
#             listb.append(x)
#     elif len(lista) == len(listb):
#         x = 0
#     else:
#         x = random.choice(elementadd)
#         while x not in lista:
#             lista.append(x)



def float_to_int_no_repeat(floatlist,listlen):
    """Change list of float numbers to list of integers with no repeat numbers"""
    intlist = []
    for float in floatlist:

        a = int(float)
        b = listlen - a
        if a not in intlist and b not in intlist:
            intlist.append(a)
        else:
            while a in intlist:
                a += 1
            intlist.append(a)
    return intlist



class WOATrainer(Trainer):

    def __init__(self, grid, outputFile,min_class_filter_num, kernel,C,gamma,  max_iterations,stop_steps,sample_size,penalty,metrics):
        self.ensemble_sample = []
        self.updated_ensemble_sample = []
        self.best_ensemble = []
        self.best_g_mean = -1
        self.best_m_auc = -1
        self.best_accuracy = []
        self.max_iterations = max_iterations
        self.sample_size = sample_size
        self.b = 1
        self.numofsample = 30
        self.localopt = [] 
        self.globalopt = []
        self.best_iteration = 0
        self.best_ensemble_iteration = -1
        self.stop_steps = stop_steps
        self.All_alternative_ensemble_solutions = [] 
        self.iteration_num = 0
        super(WOATrainer,self).__init__(grid,outputFile,min_class_filter_num,kernel,C,gamma)
        self.revise_choseone = 0.4  
        self.penalty = penalty
        self.metrics = metrics
        self.label_proba = None

    def initialization(self, num_of_sample):
        """produce num_of_sample(fixed) processed woasvmtrainer with training data,put them into list self.ensemble_sample"""
        print("AOW start initialization")
        self.numofsample = max(num_of_sample,1)
        print(self.numofsample)
        self.ensemble_sample = []
        for i in range(0, self.numofsample):
            flag = False
            while not flag:
                woasvmtrainer = self.initialChooseOne()
                flag = self.checkIdentical(woasvmtrainer)
            woasvmtrainer.completeData(self.train_data, self.train_label)
            # psosvmtrainer.train_id=self.train_id
            self.ensemble_sample.append(woasvmtrainer)
        # for j in range(1, len(self.class_label) + 1):
        #     a = np.array([i for i, x in enumerate(self.train_label) if x == j])
        #     self.class_label_classify.append(a) 
        # for i in range(0,len(self.train_data)):
        #     if self.train_label[i]==list(self.class_amount.keys())[0] or self.train_label[i]==list(self.class_amount.keys())[1]:
        #         self.Min_2_of_alternatives.append(self.train_id[i])
        print("WOA initialization complete")

    def trainWOASVM(self):
        print("WOA  start baseline training and testing")
        self.outputFileStream.write("WOA training process\n")
        for i in range(0, self.max_iterations):
            self.iteration_num = i
            self.outputFileStream.write("WOA  iteration " + str(i) + "\n")
            print("WOA iteration", i, "start training")
            self.trainByStep()  
            total_num, correct_num, accuracy, g_mean, m_auc = self.validate() 
            gmean = self.ensemble_sample[0].g_mean
            mauc = self.ensemble_sample[0].m_auc
            # if self.count_j>99:

            if self.metrics == "gmean":
                # gmean = self.ensemble_sample[0].g_mean
                for j in range(0, len(self.ensemble_sample)):  
                    if self.ensemble_sample[j].g_mean >= gmean:
                        gmean = self.ensemble_sample[j].g_mean
                        self.localopt = self.ensemble_sample[j].train_id
            elif self.metrics == "mauc":
                # mauc = self.ensemble_sample[0].m_auc
                for j in range(0, len(self.ensemble_sample)):  
                    if self.ensemble_sample[j].m_auc >= mauc:
                        mauc = self.ensemble_sample[j].m_auc
                        self.localopt = self.ensemble_sample[j].train_id

            # for j in range(0, len(self.ensemble_sample)):
            #     if (self.ensemble_sample[j].g_mean >gmean and self.ensemble_sample[j].m_auc >= 0.95*mauc)or (self.ensemble_sample[j].m_auc >mauc and self.ensemble_sample[j].g_mean >= 0.95*gmean):
            #         gmean = self.ensemble_sample[j].g_mean
            #         mauc = self.ensemble_sample[j].m_auc
            #         self.localopt = self.ensemble_sample[j].train_id
            print("Now local best index sample", self.localopt, "\n")
            # print("计数", self.count_j, "\n")
            print("Now global best index sample",self.globalopt, "\n")
            print("local best gmean and mauc,", [gmean,mauc],"\n")

            print("WOA iteration", i, "start validating")
            self.outputFileStream.write("Result on validation set:\n")
            self.outputFileStream.write("\tTotal\tCorrect\tAccuracy\n")
            correct = 0
            for k in range(0, self.classes):
                self.outputFileStream.write(
                    str(self.class_label[k]) + "\t" + str(total_num[k]) + "\t" + str(correct_num[k]) + "\t" + str(
                        accuracy[k]) + "\n")
                correct += correct_num[k]
            self.outputFileStream.write(
                "Overall\t" + str(self.validate_data.shape[0]) + "\t" + str(correct) + "\t" + str(
                    float(correct) / float(self.validate_data.shape[0])) + "\n")
            self.outputFileStream.write("G-Mean:" + str(gmean) + "\n")
            self.outputFileStream.write("mAUC:" + str(mauc) + "\n")
            print("G-mean:", gmean)
            print("mAUC:", mauc)
            if self.metrics == "gmean":
                if gmean >= self.best_g_mean:
                    self.best_g_mean = gmean
                    self.best_m_auc = mauc
                    self.best_ensemble_iteration = i
                    self.best_ensemble = self.ensemble_sample
                    self.globalopt = self.localopt
                    self.All_alternative_ensemble_solutions.append(self.best_ensemble)
                elif i - self.best_ensemble_iteration >= self.stop_steps:
                    self.outputFileStream.write("The best iteration has not been refreshed for " + str(
                        self.stop_steps) + " iterations, stop training\n")
                    print("The best iteration has not been refreshed for", self.stop_steps, "iterations, stop training")
                    break
            elif self.metrics == "mauc":
                if mauc >= self.best_m_auc:
                    self.best_g_mean = gmean
                    self.best_m_auc = mauc
                    self.best_ensemble_iteration = i
                    self.best_ensemble = self.ensemble_sample
                    self.globalopt = self.localopt
                    self.All_alternative_ensemble_solutions.append(self.best_ensemble)
            # if (gmean > self.best_g_mean and mauc >= 0.95*self.best_m_auc) or (mauc > self.best_m_auc and gmean >= 0.95*self.best_g_mean):
            #     self.best_g_mean =gmean
            #     self.best_m_auc = mauc
            #     self.globalopt = self.localopt
            #     self.best_ensemble_iteration = i
            #     self.best_ensemble = self.ensemble_sample
            #     self.All_alternative_ensemble_solutions.append(self.best_ensemble)
                elif i - self.best_ensemble_iteration >= self.stop_steps:
                    self.outputFileStream.write("The best iteration has not been refreshed for " + str(
                    self.stop_steps) + " iterations, stop training\n")
                    print("The best iteration has not been refreshed for", self.stop_steps, "iterations, stop training")
                    break
            self.outputFileStream.write("Current best iteration:" + str(self.best_ensemble_iteration) + "\n")
            self.outputFileStream.write("G-Mean on validation set:" + str(self.best_g_mean) + "\n")
            self.outputFileStream.write("mAUC on validation set:" + str(self.best_m_auc) + "\n")
            print("Current best iteration:", self.best_ensemble_iteration)
            print("G-Mean on validation set:", self.best_g_mean)
            print("mAUC on validation set:", self.best_m_auc)
            print("New iteration")
            # self.calNewPheromone()
            # self.updatePosition()
            #根据WOA更新性的index list

            self.updated_ensemble_sample=self.WOA(self.max_iterations,20, self.globalopt,i,self.ensemble_sample)
            #  updated_ensemble_sample indexout of range 
            self.ensemble_sample = self.check_and_revise(self.updated_ensemble_sample)
            print("Choose samples for next iteration")
            #self.choose()
            self.ensemble_sample = []
            for i in range(0, self.numofsample):
                flag = False
                while not flag:
                    woasvmtrainer = self.initialChooseOne()
                    flag = self.checkIdentical(woasvmtrainer)
                woasvmtrainer.completeData(self.train_data, self.train_label)
                # psosvmtrainer.train_id=self.train_id
                self.ensemble_sample.append(woasvmtrainer)
        self.outputFileStream.write("Best iteration:" + str(self.best_ensemble_iteration) + "\n")
        self.outputFileStream.write("G-Mean on validation set:" + str(self.best_g_mean) + "\n")
        self.outputFileStream.write("mAUC on validation set:" + str(self.best_m_auc) + "\n")
        print("Current best iteration:", self.best_ensemble_iteration)
        print("G-Mean on validation set:", self.best_g_mean)
        print("mAUC on validation set:", self.best_m_auc)
        # print("self.train_id_test",self.train_id_test)

    def WOA(self,max_iterations, population_size, best_solution, t,population):
        """Accordng to the best_solution, we create new population(ensemble sample) by updating the whale(list)
        and add whales to population to produce new ensemble sample"(stands for the index), need to check if there
        are indexs out of rang"""
        # Step 1: Initialize the whales population
        # population = np.zeros((population_size, 2))  # Each whale has 2 dimensions: C and gamma
        # t = 0
        # for i in range(population_size):
        #     population[i] = [random.uniform(0.1, 100), random.uniform(0.0001, 1)]

        # Initialize the best solution

        # Step 3: Main loop

        a = 2 * ((max_iterations - t) / max_iterations)
        for i in range(population_size):
            # use WOA algorithm to produce updated whale list
            whale = population[i]
            updated_whale = self.update_current_agent(best_solution,whale,a,population_size,population)
            # Step 3c: Update search agent
            # Check if any search agent goes beyond the search space and amend it
            # updated_whale = np.clip(updated_whale, [0.1, 0.0001], [100, 1])

            # Step 3d: Evaluate the fitness of the new position
            #fitness = objective_function(X, y, updated_whale[0], updated_whale[1])

            #  # Step 3e: Update the best solution if the fitness improves
            # if fitness > best_fitness:
            #     best_solution = updated_whale.copy()
            #     best_fitness = fitness
            # print(f' During {t} generation, the best Gmean is, the best mAUC is')
            population[i] = updated_whale

        return population

    def update_current_agent(self,best_solution,whale,a,population_size,population):
        """WOA1 to update"""
        # Step 3a: Update parameters a, A, C, l, and p
        emptysamplevector = np.full(self.train_label.shape[0],0)
        A = 2 * a * random.random() - a #exploration beharvior
        C = 2 * random.random()#exploitation behavior
        l = (a - 1) * random.random() + 1
        p = random.random()
        b = 1
        whaleid = np.where(whale.sample_vector == 1)[0]
        # Step 3b: Update current agent
        if p < 0.5:
            if abs(A) < 1:
                # Eq. (1): Update current agent
                D1 = []  # return updated_whale
                # revise_list_to_same_len(best_solution, whaleid,list(range(len(emptysamplevector))))
                for m in range(min(len(whaleid),len(best_solution))):
                    # D = [] #distance
                    # D.append(C*best_solution[m] -whale[m])

                    D1.append(best_solution[m] - A*(C*best_solution[m] -whaleid[m]))
                #D = abs(C * best_solution - whale)
                #return best_solution[n] - A * D[n] (for n in range(len(best_solution)))
                D1 = float_to_int_no_repeat(D1,len(emptysamplevector))
                whale.sample_vector = change_back(emptysamplevector,D1)
                return whale
            else:
                # Eq. (7): Select random agent and update current agent
                rand_whale_index = random.randint(0, population_size - 1)
                rand_whale1 = population[rand_whale_index]
                rand_whaleid = np.where(rand_whale1.sample_vector == 1)[0]
                # revise_list_to_same_len(rand_whaleid,whaleid,list(range(len(emptysamplevector))))
                D3 = []

                for n in range(min(len(whaleid),len(rand_whaleid))):

                    D2 = []
                    #D = abs(C * rand_whale - whale)
                    D2.append(C*rand_whaleid[n] - whaleid[n])
                    D3.append(rand_whaleid[n] - A*(C*rand_whaleid[n] -whaleid[n]))
                D3 = float_to_int_no_repeat(D3,len(emptysamplevector))
                rand_whale1.sample_vector = change_back(emptysamplevector,D3)
                return  rand_whale1
        else:
            # Eq. (5): Update current agent
            D5 = []
            # revise_list_to_same_len(best_solution, whaleid,list(range(len(emptysamplevector))))
            for p in range(min(len(whaleid),len(best_solution))):

                D5.append(abs(best_solution[p] - whaleid[p])*np.exp(b*l)*np.cos(2*np.pi * l ) + best_solution[p])
           # D = abs(best_solution - whale)
            D5 = float_to_int_no_repeat(D5,len(emptysamplevector))
            whale.sample_vector = change_back(emptysamplevector,D5)
            return whale
           # return  D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution


    def check_and_revise(self,updateversion):
        """To check the updated_ensemble_sample has index out of range, and revise them"""
        all_node_list = list(range(0, len(self.train_data)))
        finalre = []
        for sublistid in range(len(updateversion)):
            sublist = updateversion[sublistid].sample_vector
            #print(sublist)
            updatesublist = []
            xsub = np.where(sublist == 1)[0]
            #print(len(xsub))
            for indexid in range(len(xsub)):
                tocheck = xsub[indexid]
                if np.isin(tocheck, all_node_list):
                    #and not np.isin(xsub[indexid],updatesublist):
                    if not np.isin(tocheck,updatesublist):
                        updatesublist.append(tocheck)
                    else:
                        xlist =[x for x in all_node_list if x not in updatesublist]
                        updatesublist.append(random.choice(xlist))

                else:
                    xlist =[x for x in all_node_list if x not in updatesublist]
                    updatesublist.append(random.choice(xlist))
                #finalre.append(updatesublist)
            #print("check and revise checkk")
            #print(updatesublist)
            updateversion[sublistid].sample_vector = change_back(np.full(self.train_label.shape[0],0), updatesublist)
            return updateversion


    def choose(self):
        # self.outputFileStream.write("Print out label\n")
        # self.outputFileStream.write(str(self.train_label.tolist()))
        # self.outputFileStream.write("Print out ants\n")
        for i in range(0, self.ant):
            # self.outputFileStream.write("Ant "+str(i)+"\n")
            flag = False
            while not flag:
                woasvmtrainer = self.chooseOne()
                woasvmtrainer = self.revise(woasvmtrainer)
                flag = self.checkIdentical(woasvmtrainer)
            # self.outputFileStream.write(str(acosvmtrainer.sample_vector.tolist()))
            # self.outputFileStream.write("\n")
            woasvmtrainer.completeData(self.train_data, self.train_label)
            self.ensemble_sample.append(woasvmtrainer)


    def checkIdentical(self, woasvmtrainer):
        for trainer in self.ensemble_sample:
            if (woasvmtrainer.sample_vector == trainer.sample_vector).all():
                return False
        return True

    def getResult(self):
        print("woa training complete, start testing")
        total_num, correct_num, accuracy, g_mean, m_auc = self.test()
        self.outputFileStream.write("Result on test set:\n")
        self.outputFileStream.write("\tTotal\tCorrect\tAccuracy\n")
        correct = 0
        for k in range(0, self.classes):
            self.outputFileStream.write(str(self.class_label[k])+"\t"+str(total_num[k])+"\t"+str(correct_num[k])+"\t"+str(accuracy[k])+"\n")
            correct += correct_num[k]
        self.outputFileStream.write("Overall\t"+str(self.test_data.shape[0])+"\t"+str(correct)+"\t"+str(float(correct)/float(self.test_data.shape[0]))+"\n")
        self.outputFileStream.write("G-Mean:"+str(g_mean)+"\n")
        self.outputFileStream.write("mAUC:"+str(m_auc)+"\n")
        print("G-mean:", g_mean)
        print("mAUC:", m_auc)
        print("woa Testing finished\n")
        self.outputFileStream.close()

    def trainByStep(self):
        for e_sample in self.ensemble_sample:
            if self.grid:
                e_sample.gridTrain()
            else:
                e_sample.train(self.kernel, self.C, self.gamma)

    def validate(self):
        for e_sample in self.ensemble_sample:
            e_sample.validate(self.validate_data, self.validate_label)
        predicted, label_proba = self.predict_vote(self.ensemble_sample, self.validate_data)
        total_num, correct_num, accuracy, g_mean, m_auc = getReport(self.classes, self.class_label, predicted,
                                                                    label_proba, self.validate_label)
        return total_num, correct_num, accuracy, g_mean, m_auc

    def test(self):
        predicted, label_proba = self.predict_vote(self.best_ensemble, self.test_data)
        total_num, correct_num, accuracy, g_mean, m_auc = getReport(self.classes, self.class_label, predicted,
                                                                    label_proba, self.test_label)
        return total_num, correct_num, accuracy, g_mean, m_auc

    def predict_pro(self, ensemble, data):
        final_proba = np.zeros((data.shape[0], self.classes))
        for e in ensemble:
            if e.g_mean < 0.5:
                continue
            probas = e.clf.predict_proba(data)
            if not probas.shape[1] == self.classes:
                temp = np.zeros((1,data.shape[0]))
                delta = self.classes - probas.shape[1]
                current_classes = e.clf.classes_
                for k in range(0, len(self.class_label)):
                    if not self.class_label[k] in current_classes:
                        probas = np.insert(probas, k, values=temp, axis=1)
                        delta -= 1
                        if delta == 0:
                            break
            final_proba += probas * np.array(e.accuracy)

        def find_label(index):
            return np.array(self.class_label)[index]
        return find_label(np.argmax(final_proba, axis=1)), final_proba

    def predict_vote(self, ensemble, data):
        final_vote = np.zeros((data.shape[0], self.classes))
        used = 0
        for e in ensemble:
            #if e.g_mean < 0.5:
                #continue
            used += 1
            label = e.clf.predict(data)
            def find_index(label):
                return self.class_label.index(label)
            vfunc = np.vectorize(find_index)
            label = vfunc(label)
            label_one_hot = np.eye(self.classes)[label]
            final_vote += label_one_hot
        label_proba = final_vote / float(used)
        def find_label(index):
            return np.array(self.class_label)[index]
        return find_label(np.argmax(final_vote, axis=1)), label_proba


    def initialChooseOne(self):
        woasvmtrainer = WOASVMTrainer(self.classes, self.class_label)
        woasvmtrainer.sample_vector = np.full(self.train_label.shape[0], 0)
        self.indexes_train_data_id = []
        for l in self.class_label:
            temp = np.where(self.train_label == l)
            ##size = int(self.revise_choseone * self.sample_size * self.min_class_amount)
            indexes = np.random.choice(temp[0], int(self.revise_choseone * self.sample_size * self.min_class_amount), replace=False)
            self.indexes_train_data_id=np.append(self.indexes_train_data_id, indexes)
            woasvmtrainer.sample_vector[indexes] = 1

        #self.train_id = np.concatenate((self.indexes_train_data_id,self.train_id),axis=0)
        return woasvmtrainer


class WOASVMTrainer():
    def __init__(self, classes, class_label):
        self.sample_vector = None
        self.train_data = None
        self.train_label = None
        self.train_id = None
        self.clf = None
        self.classes = classes
        self.class_label = class_label
        self.accuracy = []
        self.g_mean = -1
        self.m_auc = -1

    def completeData(self, data, label):
        indexes = []
        for i in range(0, len(self.sample_vector)):
            if self.sample_vector[i] == 1:#sample_vector是initchose传过来的
                indexes.append(i)
        #print(len(indexes))
        self.train_id = indexes
        self.train_data = data[indexes, :]
        self.train_label = label[indexes]

    def train(self, kernel, C, gamma):
        if kernel == 'rbf':
            self.clf = SVC(gamma=gamma, C=C, probability=True)
        elif kernel == 'linear':
            self.clf = SVC(kernel='linear', C=C, probability=True)

        self.clf.fit(self.train_data, self.train_label)

    def gridTrain(self):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        self.clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='balanced_accuracy', n_jobs=-5)
        self.clf.fit(self.train_data, self.train_label)

    def validate(self, data, label):
        predicted = self.clf.predict(data)
        predicted_proba = self.clf.predict_proba(data)
        if not predicted_proba.shape[1] == self.classes:
            temp = np.zeros((1,data.shape[0]))
            delta = self.classes - predicted_proba.shape[1]
            current_classes = self.clf.classes_
            for k in range(0, len(self.class_label)):
                if not self.class_label[k] in current_classes:
                    probas = np.insert(predicted_proba, k, values=temp, axis=1)
                    predicted_proba = probas
                    delta -= 1
                    if delta == 0:
                        break
        total_num, correct_num, self.accuracy, self.g_mean, self.m_auc = getReport(self.classes, self.class_label, predicted, predicted_proba, label)

def getReport(classes, class_label, predicted_label, predicted_proba, actual_label):
    total_num = [0] * classes
    correct_num = [0] * classes
    accuracy = [0] * classes
    g_mean_product = 1.0
    for i in range(0, actual_label.shape[0]):
        actual = actual_label[i]
        actual_index = class_label.index(actual)
        total_num[actual_index] += 1
        if predicted_label[i] == actual:
            correct_num[actual_index] += 1
    for k in range(0, classes):
        accuracy[k] = float(correct_num[k])/ float(total_num[k])
        g_mean_product *= accuracy[k]
    g_mean = g_mean_product ** (1/float(len(total_num)))
    #print(classes, actual_label, predicted_proba)
    if classes > 2:
        m_auc = metrics.roc_auc_score(actual_label, predicted_proba, multi_class='ovo')
    else:
        m_auc = metrics.roc_auc_score(actual_label, predicted_label, multi_class='ovo')
    return total_num, correct_num, accuracy, g_mean, m_auc

















