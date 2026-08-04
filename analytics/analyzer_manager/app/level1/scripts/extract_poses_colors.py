import webcolors
from skimage.draw import polygon, line
import cv2
import torch
import numpy as np
from webcolors import rgb_to_name
import time
import torch.nn as nn
from level1.scripts.color_dict import *

def KMeans(x, K=21, Niter=10):
    N, D = x.shape  # Number of samples, dimension of the ambient space
    K = min(N,K)
    c = x[::int(N/K), :].clone()[:K,:]  # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    color_part = torch.zeros(K)
    for i in range(len(c)):
        color_part[i] = (cl==i).sum()/len(cl)

    c = c[color_part > 0, :]
    color_part = color_part[color_part > 0]

    return c, color_part.tolist()


def filter_color(color_name):
    
    color_name = color_name.replace('silver', 'grey')
    color_name = color_name.replace('gainsboro', 'grey')
    color_name = color_name.replace('slate', '')
    color_name = color_name.replace('deep', '')
    color_name = color_name.replace('dim', '')
    color_name = color_name.replace('forest', '')
    color_name = color_name.replace('floralwhite', '')
    color_name = color_name.replace('ghost', '')
    color_name = color_name.replace('gray', 'grey')
    color_name = color_name.replace('hot', '')
    color_name = color_name.replace('blush', '')
    color_name = color_name.replace('lawn', '')
    color_name = color_name.replace('sea', '')
    color_name = color_name.replace('chiffon', '')
    color_name = color_name.replace('sky', '')
    color_name = color_name.replace('medium', '')
    # color_name = color_name.replace('mint', '')
    color_name = color_name.replace('navajo', '')
    # color_name = color_name.replace('pale', '')
    color_name = color_name.replace('sky', '')
    color_name = color_name.replace('spring', '')
    color_name = color_name.replace('steel', '')
    color_name = color_name.replace('sandy', '')
    return color_name


class extract_poses_colors(nn.Module):
    def __init__(self, device = torch.device('cuda')):
        super().__init__()
        self.device = device
        self.color_values = None
        self.color_names = None
        self.construct_color_dic()

    def construct_color_dic(self):
        webcolors_colors = webcolors.CSS3_HEX_TO_NAMES
        self.color_values = torch.zeros((len(webcolors_colors), 3), device = self.device)
        self.color_names = []
        count = 0
        for key, name in webcolors_colors.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            self.color_values[count, 0] = r_c
            self.color_values[count, 1] = g_c
            self.color_values[count, 2] = b_c
            self.color_names.append(name)
            count += 1

    def get_colour_name(self, requested_colour):
        try:
            name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            name = self.closest_colour(requested_colour)
        return name

    
    def filter_and_get_names(self, color_part, colors):
        colors_names = []
        new_color_parts = []
        for i, color in enumerate(colors):
            closest_name = self.get_colour_name(color*255)

            #closest_name_list = [filter_color(closest_name)]
            closest_name_list = color_dictionary[closest_name]
            for c in closest_name_list:
                try:
                    name_indx =  colors_names.index(c)
                    new_color_parts[name_indx] += color_part[i]
                except:
                    colors_names.append(c)
                    new_color_parts.append(color_part[i])

        #   Filter if really small part
        colors_names_outptut = []
        for i, color_name in enumerate(colors_names):
            if new_color_parts[i] > 0.02:
                colors_names_outptut.append(color_name)

        return colors_names_outptut

    def polygon_image_limits(self, coords, image):    #   Find polygon in image limits
        _, h, w  = image.shape
        y_pol, x_pol = polygon(coords[:, 1], coords[:, 0])

        y_pol[y_pol < 0] = 0.
        y_pol[y_pol > h - 1] = h - 1

        x_pol[x_pol < 0] = 0.
        x_pol[x_pol > w - 1] = w - 1

        return y_pol, x_pol
    def find_pants_colors(self, pants_points, image):

        mask = pants_points[:,0] > 0    #   check available keypoints
        if mask.sum() == 0:    #   no keypoints in shirt
            return None
        
        if mask[0] and mask[1]: #   complete left leg available
            y_l_leg, x_l_leg = line(pants_points[0, 1], pants_points[0, 0], pants_points[1, 1], pants_points[1, 0])
            left_pants_colors = image[:,y_l_leg, x_l_leg].T
            pants_colors_l, color_part_l = KMeans(left_pants_colors)
            left_color_names = self.filter_and_get_names(color_part_l, pants_colors_l)
        elif mask[0] or mask[1]:    #   one keypoint in left leg available
            ind = 0 if mask[0] else 1
            left_color = image[:, pants_points[ind, 1], pants_points[ind, 0]]
            left_color_names = [color_dictionary[self.get_colour_name(left_color*255)]]
        else:
            left_color_names = []
            
        if mask[2] and mask[3]: #   complete right leg available
            y_r_leg, x_r_leg = line(pants_points[2, 1], pants_points[2, 0], pants_points[3, 1], pants_points[3, 0])
            right_pants_colors = image[:,y_r_leg, x_r_leg].T
            pants_colors_r, color_part_r = KMeans(right_pants_colors)
            right_color_names = self.filter_and_get_names(color_part_r, pants_colors_r)
        elif mask[2] or mask[3]:    #   one keypoint in right leg available
            ind = 2 if mask[2] else 3
            right_color = image[:, pants_points[ind, 1], pants_points[ind, 0]]
            right_color_names = [color_dictionary[self.get_colour_name(right_color*255)]]
        else:
            right_color_names = []

        for r_name in right_color_names:
            if r_name not in(left_color_names):
                left_color_names.append(r_name)

        return left_color_names


    def find_shirt_colors(self, shirt_points, image):
        valid_shirt_points = shirt_points[shirt_points[:,0] > 0, :] #   leave only found keypoints

        if len(valid_shirt_points) < 3:    #   No keypoints in shirt
            return None

        y_pol, x_pol = self.polygon_image_limits(valid_shirt_points, image)
        all_shirt_colors = image[:,y_pol, x_pol].T
        shirt_colors, color_part = KMeans(all_shirt_colors)
        output = self.filter_and_get_names(color_part, shirt_colors)
        return output


    def find_pose_colors(self,pose, image):
        kps = pose.keypoints
        shirt_points = np.array([kps[5], kps[11], kps[8], kps[2], kps[1]], dtype = np.int32)    #   l_sho, l_hip, r_hip, r_sho, neck
        pants_points = np.array([kps[11], kps[12], kps[8], kps[9]], dtype = np.int32)    #   l_hip, l_knee, r_hip, r_knee
        
        shirt_colors = self.find_shirt_colors(shirt_points, image)
        pants_colors = self.find_pants_colors(pants_points, image)

        pose.shirt_colors = shirt_colors
        pose.pants_colors = pants_colors
        return pose

    def closest_colour(self, requested_colour):
        requested_colour = requested_colour.unsqueeze(0)
        diff = requested_colour - self.color_values
        diff = (diff**2).sum(axis = 1)
        argmin = torch.argmin(diff)
        return self.color_names[argmin]
    
    def forward(self, names_space_list_1d):
        names_space_dic_1d = {}
        for c in range(len(names_space_list_1d)):  #   crop in image
            names_space_dic_1d[int(names_space_list_1d[c].id)] = []
            for p_num in range(len(names_space_list_1d[c].poses)): #   pose in crop
                names_space_dic_1d[int(names_space_list_1d[c].id)].append(self.find_pose_colors(names_space_list_1d[c].poses[p_num], names_space_list_1d[c].image[0]))

        return names_space_dic_1d        
