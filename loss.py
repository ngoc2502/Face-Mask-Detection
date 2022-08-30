from collections import namedtuple
import numpy as np

def iou_loss(predic,groundT):
    '''
    Intersection over the union between prediction and ground true 
    '''
    x1=max(predic[0],groundT[0])
    y1=max(predic[1],groundT[1])
    x2=min(predic[2],groundT[2])
    y2=min(predic[3],groundT[3])

    intersection=max(0,x2-x1+1)*max(0,y2-y1+1)

    area_prediction=(predic[2]-predic[0]+1)*(predic[3]-predic[1]+1)
    area_groundT=(groundT[2]-predic[0]+1)*(predic[3]-predic[1]+1)

    iou= intersection/area_groundT+area_prediction-intersection

    return iou



