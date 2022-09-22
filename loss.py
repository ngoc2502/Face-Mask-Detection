import numpy as np

class loss:

    def __init__(self,batch,predic,groundT):
        self.batch=batch
        self.predic=predic
        self.groundT=groundT
        self.Liou=np.zeros(batch)

    def iou_step(self,predic,GroundT):

        x1=predic[0]
        y1=predic[1]
        w1=predic[2]-predic[0]
        h1=predic[3]-predic[1]
        x2=GroundT[0]
        y2=GroundT[1]
        w2=GroundT[2]-GroundT[0]
        h2=GroundT[3]-GroundT[1]

        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
        # No overlap
        if w_intersection <= 0 or h_intersection <= 0: 
            return 0.
        I = w_intersection * h_intersection
        U = w1 * h1 + w2 * h2 - I 
        iou= I/U
        return iou

    def iou_loss(self):
        iou=np.zeros(self.batch)
        for i in range(self.batch):
            iou[i]=self.iou_step(self.predic[i],self.groundT[i])
        self.Liou=iou
        return iou

    def grad_iou(self):
        pass

    def mean_binary_loss(self):        
        count=0
        for i in range(self.batch):
            if self.predic[:][i][4]!=self.groundT[:][i][4]:
                count+=1
        return count/self.batch*100

GroundT=np.array([
    [0,0,10,10,1],
    [2 ,4 ,5 ,10,1],
    [0,0,1,1,1],
    [0,0,1,1,1],
])
predic1=np.array([
    [5 ,5 ,15 ,15,1],
    [2,4,15,15,0],
    [2,2,4,4,1],
    [0,0,1,1,1],
])

L=loss(4,predic1,GroundT)
Lbin=L.mean_binary_loss()
print(Lbin)

#               TEST IOU LOSS
# GroundT=np.array([
#     [0,0,10,10],
#     [2 ,4 ,5 ,10],
#     [0,0,1,1],
#     [0,0,1,1]
# ])
# predic1=np.array([
#     [5 ,5 ,15 ,15],
#     [2,4,15,15],
#     [2,2,4,4],
#     [0,0,1,1]
# ])
# G=np.array([
#     5,5,20,15
# ])
# P=np.array([
#     5,10,10,15
# ])

# P1=np.array([
#     [0,0,10,10]
# ])
# G1=np.array([
#     [5,5,15,15]
# ])

# np.random.seed(1)
# predic=np.random.randint(10,size=(5,4))
# L=loss(4,predic1,GroundT)
# Liou2=L.iou_loss()
# print(GroundT)
# print(predic1)
# print(Liou2)