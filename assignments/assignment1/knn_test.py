import random
from cs231n.data_utils import load_CIFAR10
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor    



def time_function(f, *args):
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

def matrix_compare(a,b):
    difference = np.linalg.norm(a-b,ord='fro')
    print "Difference was: %f" %(difference,)
    if difference < 0.001:
       print "Good! The distance matrices are the same"
    else:
       print "Uh-ohh! they are difference"
    

def cross_validate(X_train, y_train):
    num_folds = 5
    k_choices = [1,3,5,8,10,12,15,20,50,100]
    X_train_folds = []
    y_train_folds = []
    N = len(X_train)
    train_folds = np.array_split(range(N),num_folds,axis=0)
    k_to_accuracies = {}
    for k1 in k_choices:
        fold_eval = []
        for i in range(num_folds):
            mask = np.ones(N,dtype=bool)
            mask[train_folds[i]] = False
            X_train_cur = X_train[mask]
            y_train_cur = y_train[mask]
            classifier = KNearestNeighbor()
            classifier.train(X_train_cur, y_train_cur)
            
            X_test_cur = X_train[train_folds[i]]
            y_test_cur = y_train[train_folds[i]]
            
            dists = classifier.compute_distances_no_loops(X_test_cur)
            y_test_pred = classifier.predict_labels(dists,k=k1)
            num_correct = np.sum(y_test_pred == y_test_cur)
            accuracy = float(num_correct)/len(y_test_cur)
            fold_eval.append(accuracy)
            #pass
        k_to_accuracies[k1] = fold_eval[:]
        #k_to_accuracies[k1] = [1,2,3,4,5]

    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print 'k = %d, accuracy = %f' % (k, accuracy)
    
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k]*len(accuracies), accuracies)

    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.savefig('./figures/validation_k')

def test1():
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    print 'Training data shape:', X_train.shape
    print 'Training label shape:', y_train.shape
    print 'Test data shape:', X_test.shape
    print 'Test label shape:', y_test.shape

    # classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    # num_classes = len(classes)
    # sample_per_class = 7

    # for y,cls in enumerate(classes):
    #     idxs = np.flatnonzero(y_train == y)
    #     idxs = np.random.choice(idxs, sample_per_class, replace=False)
    #     for i, idx in enumerate(idxs):
    #         plt_idx = i*num_classes + y + 1
    #         plt.subplot(sample_per_class, num_classes, plt_idx)
    #         plt.imshow(X_train[idx].astype('uint8'))
    #         plt.axis('off')
    #         if i == 0:
    #             plt.title(cls)

    # plt.savefig("./figures/cifar_sample.png")
    # plt.show()
    # plt.close()

    num_training = 5000
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    num_test = 500
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0],-1))
    X_test = np.reshape(X_test,(X_test.shape[0],-1))
    print X_train.shape, X_test.shape

    from cs231n.classifiers import KNearestNeighbor    
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    

    # two_loop_time = time_function(classifier.compute_distances_two_loops,X_test)
    # print "two loop time %f" % two_loop_time

    # one_loop_time = time_function(classifier.compute_distances_one_loop,X_test)
    # print "one loop time %f " %one_loop_time
    
    # no_loop_time = time_function(classifier.compute_distances_no_loops,X_test)
    # print "no loop time %f "% no_loop_time
    
    dists = classifier.compute_distances_no_loops(X_test)

    # dist_one_loop = classifier.compute_distances_one_loop(X_test)
    # dist_two_loops = classifier.compute_distances_two_loops(X_test)
    #matrix_compare(dists,dist_one_loop)
    #matrix_compare(dists,dist_two_loops)

    y_test_pred = classifier.predict_labels(dists,k=5)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct)/num_test
    print "God %d/%d correct => accuracy: %f" %(num_correct, num_test, accuracy)
    cross_validate(X_train,y_train)
        
test1()
