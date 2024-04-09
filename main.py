import argparse
import time
from train import *

#def train_whole(grid, C, penalty, decay_rate, metrics, kernel='linear', gamma=1e-3, inputFile='page-blocks.data', min_class_filter_num=10, ant=-1, max_iterations=20, stop_steps=5, sample_size=0.9, mutate_poss=1e-2, rate=-1):
def train_whole(grid, C, penalty, decay_rate, metrics, kernel='linear', gamma=0.1, inputFile='car.data', min_class_filter_num=0,numberofsample=20, max_iterations=20, stop_steps=5, sample_size=0.9, mutate_poss=1e-2, rate=-1):

    inputFile = 'nursery.data'
    if grid:
        outputFile = inputFile.split('.')[0]+'_'+str(decay_rate)+'_'+metrics+"whole.out"
    else:
        outputFile = inputFile.split('.')[0]+'_'+str(C)+'_'+str(penalty)+'_'+str(decay_rate)+'_'+metrics+'.out'
    woatrainer =WOATrainer(grid, outputFile, min_class_filter_num, kernel, C, gamma,
                            max_iterations, stop_steps,
                            sample_size, penalty,metrics)
    woatrainer.loadData(inputFile)
    woatrainer.prepareData()
    if grid:
        outputFile = inputFile.split('.')[0]+'_'+"baseline_whole.out"
    else:
        outputFile = inputFile.split('.')[0]+'_'+str(C)+'_'+'baseline.out'
    # baselinetrainer = BaselineTrainer(grid, outputFile, min_class_filter_num, kernel, C, gamma)
    # baselinetrainer.all_data = psotrainer.all_data
    # baselinetrainer.train_data = psotrainer.train_data
    # baselinetrainer.train_label = psotrainer.train_label
    # baselinetrainer.test_data = psotrainer.test_data
    # baselinetrainer.test_label = psotrainer.test_label
    # baselinetrainer.validate_data = psotrainer.validate_data
    # baselinetrainer.validate_label = psotrainer.validate_label
    # baselinetrainer.classes = psotrainer.classes
    # baselinetrainer.class_label = psotrainer.class_label
    # baselinetrainer.imbalance_ratio = psotrainer.imbalance_ratio

    woatrainer.initialization(numberofsample)
    woatrainer.trainWOASVM()
    woatrainer.getResult()

    #baselinetrainer.trainSVM()
    #baselinetrainer.trainAllBaselines()

if __name__ == "__main__":
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.descrption='Please enter parameters'
    parser.add_argument("-i", "--input", help="input data file path", dest="inputFile", type=str, default="car.data")
    #parser.add_argument("-o", "--output", help="output file, if not mentioned, will print to screen", dest="outputFile", type=str, default=None)
    parser.add_argument("-min_filter", "--min_class_filter", help="the class less than this amount will be ignored", dest="min_class_filter_num", type=int, default=20)
    parser.add_argument("-grid", "--use_grid_search", help="use grid search to find the best svm", dest="grid", type=bool, default=False)
    parser.add_argument("-kernel", "--svm_kernel", help="the kernel svm classifier to use, supported:rbf, linear", dest="kernel", type=str, default="linear")
    parser.add_argument("-C", "--svm_penalty", help="the penalization parameter for svm", dest="C", type=float, default=4)
    parser.add_argument("-gamma", "--svm_gamma", help="gamma for rbf kernel svm", dest="gamma", type=float, default=0.1)
    parser.add_argument("-ant", "--ant_size", help="ant size, should more than imbalance ratio", dest="ant", type=int, default=20)
    parser.add_argument("-iter", "--max_iterations", help="max iterations for PSO", dest="max_iterations", type=int, default=20)
    parser.add_argument("-stop", "--stop_iterations_without_optimization", help="the number of iterations to stop without optimization for ants", dest="stop_steps", type=int, default=5)
    parser.add_argument("-decay", "--decay_rate", help="decay rate for pheromone", dest="decay_rate", type=float, default=0.97)
    parser.add_argument('-sapl_size', "--sample_size", help="the ratio of min class instances selected for the first iteration of pso", dest="sample_size", type=float, default=0.9)
    parser.add_argument('-mutate', "--mutation_possibility", help="the mutation possibility of the ants", dest="mutate_poss", type=float, default=1e-2)
    parser.add_argument('-pen', "--penalization", help="the penalization for unchosen sample", dest="penalty", type=float, default=1e-4)
    parser.add_argument('-rate', "--choose_rate", help="the rate multiply on 1/n_instances for the possibility choosing samples", dest="rate", type=float, default=-1)
    parser.add_argument('-metrics', "--best_metrics", help="the metrics used to get the best validation, gmean/mauc only", dest="metrics", type=str, default="mauc")
    args = parser.parse_args()

    train_whole(args.grid, args.C, args.penalty, args.decay_rate, args.metrics,
                args.kernel, args.gamma, args.inputFile, args.min_class_filter_num,
                args.ant, args.max_iterations, args.stop_steps,
                args.sample_size, args.mutate_poss, args.rate)
    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime, "秒")

