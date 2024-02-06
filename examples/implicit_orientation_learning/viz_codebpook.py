import cv2
for i in range(92000):
    image = cv2.imread('dict_images/dict_{}.png'.format(i))
    cv2.imshow('codebook_', image)
    cv2.waitKey(1)
