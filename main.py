from sklearn.cluster import KMeans
from skimage import io, filters, measure, draw
from skimage.feature import canny, corner_harris, corner_peaks
import numpy as np
import matplotlib.pyplot as plt
import cv2

#tl, tr, br, bl are top-left, top-right, bottom-right, bottom-left
#coordinates of the board rectangle
def find_stones(img, board_size, board_rect):
    (tl, tr, br, bl) = board_rect
    colors = np.zeros((board_size, board_size, 3))
    end = board_size-1
    src = np.array([(0,0), (0,end), (end,end), (end,0)], dtype="float32")
    dst = np.array([tl, tr, br, bl], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)

    for i in range(board_size):
        #approximate field width for a row
        points = np.array([[[i, 0], [i, board_size-1]]], dtype="float32")
        warped = cv2.transform(points, M)
        left  = warped[0][0] / warped[0][0][2]
        right = warped[0][1] / warped[0][1][2]
        row = (right - left)
        width = np.sqrt(row[0]**2 + row[1]**2) / (board_size-1)
        for j in range(board_size):
            #TODO(piotr): [[[ugh]]]
            p = np.array([[[i, j]]], dtype="float32")
            p = cv2.transform(p, M)
            p = p[0][0] / p[0][0][2]
            rr, cc = draw.circle(p[0], p[1], .2*width, shape=img.shape)
            r = np.average(img[rr,cc,0])
            g = np.average(img[rr,cc,1])
            b = np.average(img[rr,cc,2])
            colors[i,j] = [r,g,b]
            img[rr,cc] = [255,255,255] #DEBUG, mark the test points

    colors_flat = colors.reshape(board_size*board_size,3)
    r = np.average(colors_flat[:,0])
    g = np.average(colors_flat[:,1])
    b = np.average(colors_flat[:,2])
    init = np.array([[0.,0.,0.], [r,g,b], [1.,1.,1.]])
    groups = KMeans(n_clusters = 3, init = init, n_init = 1)
    groups.fit(colors_flat)
    labels = groups.labels_.reshape(board_size,board_size)
    avg = [np.average(groups.cluster_centers_[i]) for i in range(3)]
    black = min(enumerate(avg), key=lambda x: x[1])
    white = max(enumerate(avg), key=lambda x: x[1])

    row = ['.'] * board_size
    result = [row.copy() for i in range(board_size)]
    for i in range(board_size):
        for j in range(board_size):
            if(labels[i,j] == black[0]):
                result[i][j] = 'B'
            if(labels[i,j] == white[0]):
                result[i][j] = 'W'
    return result

def print_board(board):
    for i in range(19):
        for j in range(19):
            print(board[i][j], end=' ')
        print('')


def test_find_stones_1():
    fig, ax = plt.subplots(1, 1, figsize=(30,20))
    img = io.imread('examples/t1_006.jpg')
    (height, width, colors) = img.shape
    tl = [125, 390]
    tr = [110, width-398]
    br = [height-150, width-435]
    bl = [height-140, 455]
    board = find_stones(img, 19, (tl, tr, br, bl))
    print_board(board)
    io.imshow(img)
    points = np.array([tl, tr, br, bl, tl])
    ax.plot(points[:,1], points[:,0], marker='o', markersize=30)
    plt.savefig('res.png')

def test_find_stones_2():
    fig, ax = plt.subplots(1, 1, figsize=(30,20))
    img = io.imread('examples/t3_004.jpg')
    (height, width, colors) = img.shape
    tl = [28, 142]
    tr = [20, width-92]
    br = [height-37, width-13]
    bl = [height-72, 28]
    board = find_stones(img, 19, (tl, tr, br, bl))
    print_board(board)
    
    io.imshow(img)
    points = np.array([tl, tr, br, bl, tl])
    ax.plot(points[:,1], points[:,0], marker='o', markersize=30)
    plt.savefig('res.png')

def test_find_stones_3():
    fig, ax = plt.subplots(1, 1, figsize=(30,20))
    img = io.imread('examples/t2_100.jpg')
    (height, width, colors) = img.shape
    tl = [53, 117]
    tr = [40, width-120]
    br = [height-32, width-20]
    bl = [height-32, 18]
    board = find_stones(img, 19, (tl, tr, br, bl))
    print_board(board)
    
    io.imshow(img)
    points = np.array([tl, tr, br, bl, tl])
    ax.plot(points[:,1], points[:,0], marker='o', markersize=30)
    plt.savefig('res.png')

def test_peaks_detection():
    fig, ax = plt.subplots(1, 1, figsize=(30,20))
    img = io.imread('examples/t3_004.jpg', as_gray=True)
    img = canny(img)
    har = corner_harris(img)
    peaks = corner_peaks(har, min_distance=7)
    ax.plot(peaks[:, 1], peaks[:, 0], marker='o', linestyle='None', markersize=30)
    io.imshow(img)
    plt.savefig('res.png')

test_find_stones_3()
