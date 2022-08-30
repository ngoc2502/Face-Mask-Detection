from PIL import Image, ImageOps

def normalize(path):
    '''
    Nomalize images into grayscale image (mean /255)
    '''
    for i in range(len(path)):
        img=Image.open(path[i])
        path[i]=ImageOps.grayscale(img)

      
