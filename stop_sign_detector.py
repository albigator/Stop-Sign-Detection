'''
ECE276A WI20 HW1
Stop Sign Detector
'''
import os, cv2
from skimage.measure import label, regionprops
import numpy as np
from scipy.stats import multivariate_normal as mvn
import math
import matplotlib.pyplot as plt
import cv2 as cv

from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate

class StopSignDetector():
    def __init__(self):
        self.w = np.load('weights.npy')
        self.Sigma = np.load('sigma.npy')
        self.mean = np.load('mean.npy')
    def segment_image(self, img):        
        nr, nc, d = img.shape
        n=nr*nc
        xtest=np.reshape(img,(n,d))
        likelihoods=np.zeros((7,n))
        log_likelihood=np.zeros(n)
        for k in range(7):
            likelihoods[k] = (self.w)[k] * mvn.pdf(xtest, self.mean[k], self.Sigma[k],allow_singular=True)
            log_likelihood = likelihoods.sum(0)
        log_likelihood = np.reshape(log_likelihood, (nr, nc))
        log_likelihood[log_likelihood > np.max(log_likelihood) / 1.5] = 1
        print(log_likelihood.shape)
        mask_img = log_likelihood
        mask_img = np.array(mask_img)
        mask_img = (mask_img).astype('uint8')
        return mask_img

    def get_bounding_box(self, img):
        boxes=[]
        mask_img=self.segment_image(img)
        label_img = label(mask_img)
        regions = regionprops(label_img)
        fig, ax = plt.subplots()
        ax.imshow(label_img, cmap=plt.cm.gray)

        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            if props.bbox_area/(label_img.shape[0]*label_img.shape[1]) > .007 :
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax.plot(bx, by, '-b', linewidth=2.5)
                boxes.append([minc, label_img.shape[0]-maxr,maxc,label_img.shape[0]-minr])
        plt.show()
        # YOUR CODE HERE
        boxes.sort(key = lambda x: x[0])
        return boxes


if __name__ == '__main__':
    #folder = "trainset"
    my_detector = StopSignDetector()
    #img = cv2.imread(os.path.join(folder,filename))
    img = cv.imread('90.jpg')
    print(img.shape)
    mask_img = my_detector.segment_image(img)
    boxes = my_detector.get_bounding_box(img)
    #print(boxes)
    cv_mask = np.zeros((mask_img.shape[0],mask_img.shape[1],3))
    cv_mask[:, :, :3] = mask_img[:,:,np.newaxis]*255
    for box in boxes:
        cv2.rectangle(cv_mask, (box[0],mask_img.shape[0]-box[1]), (box[2],mask_img.shape[0]-box[3]), (0, 255, 0), 2)
        print(box)
    #cv2.imshow('mask and bounding box', cv_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    