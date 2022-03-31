from torch import threshold
import cv2
import numpy as np
from scipy.optimize import least_squares

def computeRootSIFTDescriptors(descriptor):
    descriptor /= np.linalg.norm(descriptor, ord=1, axis=1, keepdims=True)
    descriptor = np.sqrt(descriptor)
    return descriptor

def match2d(kp1, des1, kp2, des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(des1,des2,k=2)
    # matches = bf.match(des1, des2)
    # Apply ratio test
    matchLocations = []
    descriptors = []
    good = []
    mask = np.zeros(len(kp1),dtype=bool)
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            mask[m.queryIdx] = True
            good.append([m])
            img1Pt = kp1[m.queryIdx].pt
            img2Pt = kp2[m.trainIdx].pt
            descriptors.append(des1[m.queryIdx])
            matchLocations.append([img1Pt[0],img1Pt[1], img2Pt[0], img2Pt[1]])
    return np.array(matchLocations), np.array(descriptors),  good, mask

def flannMatcher(kp1, des1, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # ratio test as per Lowe's paper
    matchLocations = []
    descriptors = []
    good = []
    mask = np.zeros(len(kp1),dtype=bool)
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
            mask[m.queryIdx] = True
            img1Pt = kp1[m.queryIdx].pt
            img2Pt = kp2[m.trainIdx].pt
            descriptors.append(des1[m.queryIdx])
            matchLocations.append([img1Pt[0],img1Pt[1], img2Pt[0], img2Pt[1]])
    return np.array(matchLocations), np.array(descriptors),  good, mask

def showImage(image):
    cv2.namedWindow("title",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("title", image.shape[1]//8, image.shape[0]//8)
    cv2.imshow("title",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reproject(H, uv):
    uv2 = H@np.vstack([uv.T, np.ones(uv.shape[0])])
    return (uv2[:2,:]/uv2[2,:]).T

def estimateHomography(keyPoints):
    kp1 = keyPoints[:,:2]
    kp2 = keyPoints[:,2:]

    homography = lambda x: np.concatenate([x, [1]]).reshape(3,3)

    reprojectionError = lambda x, kp1, kp2: (kp1-reproject(homography(x),kp2)).reshape(-1)

    num_inliers = 0
    true_inliers = None
    threshold = 0.8
    for _ in range(100):
        sample = np.random.choice(kp1.shape[0], size=4,replace=False)
    
        result = least_squares(lambda x:
                    reprojectionError(x, kp1[sample,:], kp2[sample,:]),
                    np.eye(3).reshape(-1)[:-1], method="lm")
        e = reprojectionError(result.x, kp1, kp2)
        inliers = np.linalg.norm(e.reshape(-1,2),axis=1) < threshold
        if np.count_nonzero(inliers) > num_inliers:
            true_inliers = inliers
            num_inliers = np.count_nonzero(inliers)

    
    print(num_inliers, "/", kp1.shape[0], "inliers")

    result = least_squares(lambda x: 
                    reprojectionError(x, 
                    kp1[true_inliers,:],
                    kp2[true_inliers,:]),
                    np.eye(3).reshape(-1)[:-1], method="lm")

    return homography(result.x)

    

if __name__ == "__main__":
    images = []
    for i in range(1,7):
        images.append(cv2.imread(f"img{i}.jpg"))
    sift = cv2.SIFT_create()
    key_points = []
    descriptors = []
    for image in images:
        
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(image,None)
        des = computeRootSIFTDescriptors(des)
        key_points.append(kp)
        descriptors.append(des)
        print(len(kp))

    lhs_kp = key_points[0]
    lhs_descriptors = descriptors[0]
    
    width = 3*sum([image.shape[1] for image in images])//2
    height= images[0].shape[0]*3
    H = np.eye(3)
    H[1,2] = height/3
    H_list = [H]
    for i in range(1,len(images)):
        rhs_kp = key_points[i]
        rhs_descriptors = descriptors[i]
        matchLocations, _, _, mask = flannMatcher(lhs_kp, lhs_descriptors, rhs_kp, rhs_descriptors)
        H = H@estimateHomography(matchLocations)
        H_list.append(H)
        non_duplicate_lhs = tuple((lhs_kp[i] for i in range(len(lhs_kp)) if not mask[i]))
        lhs_kp =  cv2.KeyPoint.convert(reproject(np.linalg.inv(H),cv2.KeyPoint.convert(non_duplicate_lhs)))+ rhs_kp
        lhs_descriptors = np.vstack([lhs_descriptors[~mask],rhs_descriptors])
    H = np.eye(3)
    H[1,2] = height/3
    H[0,2] = width/2-images[0].shape[0]/2
    H_mid_inv = np.linalg.inv(H_list[len(H_list)//2])
    H_mid_inv /= H_mid_inv[2,2]
    H_list = [H_mid_inv@H_ for H_ in H_list]
    H_list = [H@H_ for H_ in H_list]
    image = np.zeros((height,width,3), dtype=images[0].dtype)
    for H, img in zip(H_list, images):
        print(H)
        converted = cv2.warpPerspective(img, H, (width, height))
        image[~np.any(image,2)] = converted[~np.any(image,2)]
        showImage(image)

