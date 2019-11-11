import math
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2


def hough_transform(image):
    img = cv2.imread(image,2)
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    (x,y)=bw_img.shape

    rmax=int(round(math.sqrt(x*x+y*y),0))
    mac=np.zeros((rmax,180))

    for a in range(1,y):
        for b in range(1,x):
            if img[b][a]==0:
                for m in range(1,180):
                    r=round((x*math.cos((m*math.pi)/180)+y*math.sin((m*math.pi))/180),0)
                    if r>0:
                        mac[int(r)][m]=mac[int(r)][m]+1

    res=io.imshow(mac,cmap=plt.cm.get_cmap('gray'))
    plt.axes(None)
    plt.axis('off')
    #plt.show()
    return mac
    #plt.savefig('test.png')


mac=hough_transform('examples/t2_100.jpg')
io.imshow(mac,cmap=plt.cm.get_cmap('gray'))
plt.savefig('test.png')
