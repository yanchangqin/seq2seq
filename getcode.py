import os
import numpy as np
import matplotlib.image as implt
import PIL.Image as image

path = './code'
class Imagedate:
    def __init__(self):
        self.image_dataset = []
        for filename in os.listdir(path):
            x = implt.imread(os.path.join(path, filename))
            label = filename.split('.')[0]
            y = self.one_hot(label)
            self.image_dataset.append([x,y])
            # print(self.image_dataset)

    def get_code(self,size):
        xs = []
        ys = []
        for j in range(size):
            num = np.random.randint(0,len(self.image_dataset))
            x = self.image_dataset[num][0]
            y = self.image_dataset[num][1]
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)/255-0.5
        return xs,ys

    def one_hot(self,x):
        arr = np.zeros([4,10])
        for i in range(4):
            index = int(x[i])
            arr[i][index]=1
        return arr
igd = Imagedate()
# igd.get_code(1)