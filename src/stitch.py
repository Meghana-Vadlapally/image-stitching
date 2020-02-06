# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:37:11 2019

@author: Karthik Vikram
"""
import cv2
import numpy as np
import os
import glob
import argparse

def parse_args():
    # Gets arguments from command line
    
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument(
        "directory", type=str, default="",
        help="path to the images that must be used for panaroma")
    args = parser.parse_args()
    return args

def calculate_homography(pts1, pts2):
    #Calculates the H matrix
    
    mat = np.zeros((2*pts1.shape[0],8))
    length = pts1.shape[0]
    for i in range(0,length):
        x1,y1 = pts1[i,0], pts1[i,1]
        x2,y2 = pts2[i,0], pts2[i,1]
        mat[i*2,:] = np.array([x1,y1,1,0,0,0,-(x1*x2),-(y1*x2)])
        mat[i*2+1,:] = np.array([0,0,0,x1,y1,1,-(x1*y2),-(y1*y2)])
    mat = np.array(mat)
    dst_flatten = pts2.flatten().T
    a, b, c, d, e, f, g, h = np.linalg.lstsq(mat, dst_flatten,rcond=None)[0]
    homography = np.array([[a, b, c], [d, e, f], [g, h, 1]])
    return homography

def change_to_homo_form(transformation_matches):
      #Changes the poins to homogeneous point with scale as 1 
    
      transformed_matches = np.zeros(transformation_matches.shape)
      for i in range(3):
          transformed_matches[:,i] = transformation_matches[:, i] / transformation_matches[:, 2]
      return transformed_matches
    
def ransac(matches,itera=15000):
  
  # Performs the Ransac Procedure
  best_h_mat = np.zeros((3, 3))
  maximum_inliers = 0
  threshold = 0.98
  for i in range(0, itera):
      idx = np.random.choice(len(matches), 4)
      rand_matches = matches[idx]
      matches1, matches2 = rand_matches[:, :2], rand_matches[:, 2:]
      H_mat = calculate_homography(matches2, matches1)
      transformation_points = np.ones(((matches.shape[0]), 3))
      transformation_points[:, :2] = matches[:, 2:]
      transformation_matches = np.matmul(H_mat, transformation_points.T).T
      transformed_matches = change_to_homo_form(transformation_matches)
      
      #Finding the distances between the transformed points and the first set of points
      
      ssd = ((matches[:, :2] - transformed_matches[:, :2])**2).sum(1)
      distances = np.sqrt(ssd)
      num_inliers = len(distances[distances < threshold])
      if num_inliers > maximum_inliers:
          maximum_inliers = num_inliers
          best_h_mat = H_mat
  return best_h_mat

def form_points(kp,matches):
    
    # Extracts coordinates from matches
    list_a=[]
    list_b=[]
    for i in range(len(matches)):
        list_a.append(kp[i].pt[0])
        list_b.append(kp[i].pt[1])
    return np.stack((list_a,list_b), axis=-1)

def sum_of_vector(x):
    
    # sums the elements of a vector
    sum= 0
    for i in range(len(x)):
            sum += x[i]
    return sum

def sum_mat(a):
    
    # sums the elements of a matrix
    sum=0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            sum += a[i][j]
    return sum

def match_by_diff(kp1,kp2,des1,des2,threshold=1800):
    
    # Finds the matches by finding the sum of differnece of descriptors and 
    res = []
    for i in range(len(des1)):
        for j in range(len(des2)):
            if(sum_of_vector(np.subtract(des1[i],des2[j]))<threshold):
                    res.append((kp1[i],kp2[j]))
    return res

def src_and_dst(temp_kp):
    
    # Splits the object into lists of source and destination points
    
    src_lst = []
    dst_lst = []
    for i in range(len(temp_kp)):
        src_lst.append((temp_kp[i][0].pt[0],temp_kp[i][0].pt[1]))
        dst_lst.append((temp_kp[i][1].pt[0],temp_kp[i][1].pt[1]))
    return src_lst,dst_lst                

def find_order(a1,b1,c1,a_name,b_name,c_name,thers=2000):
    # Finds the middle image then makes call to find the left and right image
    
    matches1 = len(give_matches(a1,b1,thres=thers))
    matches2 = len(give_matches(b1,c1,thres=thers))
    matches3 = len(give_matches(c1,a1,thres=thers))
    if (matches1 <= matches2) and (matches1 <= matches3): 
        left_image,right_image,middle_image,left_name,right_name,middle_name=find_left_right(a1,b1,c1,a_name,b_name,c_name,thers)
    elif (matches2 <= matches1) and (matches2 <= matches3): 
        left_image,right_image,middle_image,left_name,right_name,middle_name=find_left_right(b1,c1,a1,b_name,c_name,a_name,thers)
    else: 
        left_image,right_image,middle_image,left_name,right_name,middle_name=find_left_right(a1,c1,b1,a_name,c_name,b_name,thers)
    return left_image,right_image,middle_image,left_name,right_name,middle_name


def read_from_directory(path,thres=2000):
    # Reads all images from folders and make call to order it
    
    image_list = []
    name_list = []
    for filename in glob.glob(path): #assuming gif
        filename=filename.replace('\\','/')
        im=cv2.imread(filename)
        image_list.append(im)
        name_list.append(os.path.basename(filename))
    a,b,c=image_list[:3]
    a1 = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    b1 = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    c1 = cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
    a_name , b_name , c_name = name_list[:3]
    return find_order(a1,b1,c1,a_name,b_name,c_name,thers=2000)

def give_matches(a1,b1,thres=2000):
    # Uses ORB then calculates the matches between keypoints
    
    orb = cv2.ORB_create()
    kp_a2,des_a2=orb.detectAndCompute(a1,None)
    kp_b2,des_b2=orb.detectAndCompute(b1,None)
    matches = match_by_diff(kp_a2,kp_b2,des_a2,des_b2,threshold=thres)
    return matches


def find_left_right(a1,b1,c1,a_name,b_name,c_name,thres=2000):
    # Finds which image is left and which is right
    
    left_name = ""
    right_name = ""
    middle_name = ""
    if (sum_mat(np.subtract(a1,b1))==0):
        left_name = a_name
        right_name = b_name
        middle_name = c_name
    else:
        a2=a1[:,:int(a1.shape[1]/2)]
        b2=b1[:,int(b1.shape[1]/2):]
        matches1 = give_matches(a2,b2,thres)
        b2=b1[:,:int(b1.shape[1]/2)]
        a2=a1[:,int(a1.shape[1]/2):]
        matches2 = give_matches(b2,a2)
        if len(matches1)>=len(matches2):
            temp = a1
            a1=b1
            b1=temp
            left_name = a_name
            right_name = b_name
            middle_name = c_name
        else:
            left_name = a_name
            right_name = b_name
            middle_name = c_name
    return a1,b1,c1,left_name,right_name,middle_name

def warp_transform(image,x_min,y_min,x_max,y_max,H_mat):
    # Warps the given image using cv2 function
    
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0,0,1]])
    return cv2.warpPerspective(image, translation_matrix.dot(H_mat), (x_max - x_min, y_max - y_min))
    

def form_pano(first_image, middle_image, third_image):
    # Forms the panaroma output by stitching left transform and right transform images
    
    matches1_2=give_matches(first_image,middle_image,2000)
    matches2_3=give_matches(middle_image,third_image,2000)
    src_lst1_2,dst_lst1_2=src_and_dst(matches1_2)
    src_lst2_3,dst_lst2_3=src_and_dst(matches2_3)
    src_lst1_2 = np.array(src_lst1_2)
    dst_lst1_2 = np.array(dst_lst1_2)
    src_lst2_3 = np.array(src_lst2_3)
    dst_lst2_3 = np.array(dst_lst2_3)
    matches1_2 = np.hstack((dst_lst1_2,src_lst1_2))
    matches2_3 = np.hstack((src_lst2_3,dst_lst2_3))
    first_h = ransac(matches1_2,itera=20000)
    third_h = ransac(matches2_3,itera=20000)
    
    # Finding the boundaries of the images
    first_image_boundary = np.float32([[0,0], [0,first_image.shape[0]], [first_image.shape[1], first_image.shape[0]], [first_image.shape[1],0]]).reshape(-1,1,2)
    middle_image_boundary = np.float32([[0,0], [0,middle_image.shape[0]], [middle_image.shape[1], middle_image.shape[0]], [middle_image.shape[1],0]]).reshape(-1,1,2)
    third_image_boundary = np.float32([[0,0], [0,third_image.shape[0]], [third_image.shape[1], third_image.shape[0]], [third_image.shape[1],0]]).reshape(-1,1,2)
    middle_transformed_bd = cv2.perspectiveTransform(middle_image_boundary,first_h )
    third_transformed_bd = cv2.perspectiveTransform(third_image_boundary, third_h )
    points = np.concatenate((first_image_boundary, middle_transformed_bd,third_transformed_bd), axis=0)
    
    #Finding the left most x and y points
    mins = np.int32(points.min(axis=0).ravel())
    x_min = mins[0]
    y_min = mins[1]
    
    #Finding the right most x and y points
    maxs = np.int32(points.max(axis=0).ravel())
    x_max = maxs[0]
    y_max = maxs[1]
    
    translate_by = (-x_min, -y_min)
    
    #creating the left and right transformed images using the two homographies
    
    left_transform = warp_transform(first_image,x_min,y_min,x_max,y_max,first_h)
    right_transform = warp_transform(third_image,x_min,y_min,x_max,y_max,third_h)
    
    # pasting both images in numpy matrix
    result = left_transform + right_transform
    
    #pasting the middle image in the already formed left and right transformed image.
    result[translate_by[1]:first_image.shape[0]+translate_by[1],translate_by[0]:first_image.shape[1]+translate_by[0]] = middle_image
    return result

def main():
    args = parse_args()
    args_dir = ""
    args_dir = args.directory
    args_dir = args_dir.replace('//','\\')
    path = args_dir+"\*.jpg"
    
    left_image,right_image,middle_image,left_name,right_name,middle_name = \
    read_from_directory(path,700)

    dir=path.split("*", 1)[0]
    left_org = cv2.imread(dir+str(left_name))
    middle_org = cv2.imread(dir+str(middle_name))
    right_org = cv2.imread(dir+str(right_name))
    cv2.imwrite(os.path.join(args_dir , 'panorama.jpg'),form_pano(left_org,middle_org,right_org))
 
    
if __name__ == "__main__":
    main()
