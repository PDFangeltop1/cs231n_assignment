from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.classifiers.neural_net import *
import numpy as np
from cs231n.data_utils import load_CIFAR10

import matplotlib.pyplot as plt

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

hidden_size = [600,800,1000,1200,1400,2000]
training_epochs = [45]
regs = [1e-2,1e-3,5e-3,1e-4,5e-4,2]
learning_rate = [1e-3,1e-4,5e-5,1e-6,5e-6]
best = 0
bmodel = None
best_para = {}
for hs in hidden_size:
    for ep in training_epochs:
        for reg in regs:
            for lr in learning_rate:
                print "test on param hs :",hs," ep: ",ep,"  reg: ",reg,"  lr:",lr
                model = init_two_layer_model(32*32*3, hs, 10) # input size, hidden size, number of classes
                trainer = ClassifierTrainer()
                best_model, loss_history, train_acc, val_acc = trainer.train(X_train, y_train, X_val, y_val,
                                             model, two_layer_net,
                                             num_epochs=ep, reg=reg,
                                             momentum=0.9, learning_rate_decay = 0.95,
                                             learning_rate=lr, verbose=True)

                plt.subplot(2,1,1)
                plt.plot(loss_history)
                plt.title('Loss history with paramater hs: %d, reg: %f, lr: %lr'%(hs,reg,lr))
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                
                plt.subplot(2, 1, 2)
                plt.plot(train_acc)
                plt.plot(val_acc)
                plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
                plt.xlabel('Epoch')
                plt.ylabel('Clasification accuracy')
                
                plt.savefig('./figures/two_layer_figure_with_paramater_hs_%d_reg_%f_lr_%lr.png'%(hs,reg,lr))
                if val_acc > best:
                    best_para['hs'] = hs
                    best_para['ep'] = ep
                    best_para['reg'] =reg
                    best_para['lr'] = lr
                    best = val_acc
                    bmodel = best_model

for key in best_para.keys():
    print key," : ",best_para[key]
    
print "acc_val : ",best
scores = three_layer_net(X_test,bmodel)
y_pred_val = np.argmax(scores,axis=1)
val_acc = np.mean(y_pred_val == y_test)
print "acc_test :",val_acc    
