from __future__ import division
import numpy as np
def mce(pred, actual, threshold):
    err = 0
    cerr=0
    for i in range(len(actual)):
        if((pred[i]>=threshold and actual[i]==0)): 
            err = err + 1
        elif((pred[i]<threshold and actual[i]==1)):
            err =err + 1
            cerr = cerr+1
    return err, cerr
        
    

def eapError(pred, actual):
    err = 0
    positives = sum(actual)
    order = np.array(pred).argsort()
    ranks = order.argsort()
    for i in range(len(actual)):
        if(actual[i]==1): #If it is an actual click, then it should appear when the predictions are sorted as top prediction
            if(ranks[i]+1>=positives):
                pass
            else:
                err = err + 1
    return err
    
def F1_score(tags,pred, threshold):
    predicted = []
    #print tags
    for i in range(len(pred)):
        if(pred[i]>threshold):
            predicted.append(1)
        else:
            predicted.append(0)
    #print predicted
    tp, fp, fn, tn = 0,0,0,0
    for i in range(len(pred)):
        if(predicted[i]==1 and tags[i]==1):
            tp = tp + 1
        elif(predicted[i]==0 and tags[i]==1):
            fn = fn + 1
        elif(predicted[i]==0 and tags[i]==0):
            tn = tn + 1
        elif(predicted[i]==1 and tags[i]==0):
            fp = fp + 1
    #print tp, fp, fn
    if tp>0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)
    
        return precision, recall, 2*((precision*recall)/(precision+recall))
    else:
        return 0,0,0
    

