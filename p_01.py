import numpy as np
a = np.arange(24).reshape(2,3,4)
b = [[[ 10,1,2,3],
      [ 4,5,6,7],
      [ 8,9,10,11]],

     [[12,13,14,15],
      [16,17,18,19],
      [20,21,22,23]]]
print(a)
print(np.argmax(a,axis=0))
print((np.argmax(a)==np.argmax(b)).all())