import cv2
import nudged
import numpy as np

def align_face(im, src_pts, target_pts, dim):
    trans = nudged.estimate(src_pts, target_pts)
    T = np.float32(trans.get_matrix())
    im = cv2.warpAffine(im, T[0:2], (dim, dim))
    return im

class VGGAlign:
    """
    Use to align inputs to the VGG DNNs
    """
    dim = 224
    selected_indices = [36, 45, 33]
    target_pts = np.float32([\
                             [26,93],
                             [27,114],
                             [30,135],
                             [33,157],
                             [40,177],
                             [53,193],
                             [71,206],
                             [92,216],
                             [117,219],
                             [142,216],
                             [163,205],
                             [178,190],
                             [189,172],
                             [193,151],
                             [195,128],
                             [196,107],
                             [196,85],
                             [37,69],
                             [46,53],
                             [63,46],
                             [81,47],
                             [98,53],
                             [124,51],
                             [141,43],
                             [159,41],
                             [177,48],
                             [185,64],
                             [113,72],
                             [114,86],
                             [115,99],
                             [116,113],
                             [96,129],
                             [105,131],
                             [115,133],
                             [125,130],
                             [135,129],
                             [57,82],
                             [66,75],
                             [78,74],
                             [90,81],
                             [79,85],
                             [67,86],
                             [133,80],
                             [145,72],
                             [157,72],
                             [166,78],
                             [157,83],
                             [145,83],
                             [81,163],
                             [93,154],
                             [105,150],
                             [115,153],
                             [126,150],
                             [139,155],
                             [151,164],
                             [139,172],
                             [127,174],
                             [115,175],
                             [105,174],
                             [93,171],
                             [86,163],
                             [105,159],
                             [115,160],
                             [126,159],
                             [145,163],
                             [126,160],
                             [115,161],
                             [105,159]\
                         ])
    @staticmethod
    def align(im, src_pts):
        """
        given an image and the src points, align it
        """
        im = align_face(im,\
                        src_pts[VGGAlign.selected_indices],\
                        VGGAlign.target_pts[VGGAlign.selected_indices],\
                        VGGAlign.dim)
        return im
    
class OpenFaceAlign:
    """
        Use to align inputs to the OpenFace DNNs
    """
    dim = 96
    selected_indices = [36, 45, 33]
    target_pts = np.float32([\
                             [0,17],
                             [0,30],
                             [2,43],
                             [5,56],
                             [10,68],
                             [17,78],
                             [27,87],
                             [37,94],
                             [49,96],
                             [60,94],
                             [71,86],
                             [80,77],
                             [88,67],
                             [93,55],
                             [95,42],
                             [96,28],
                             [96,15],
                             [9,7],
                             [15,2],
                             [23,1],
                             [31,2],
                             [39,6],
                             [55,5],
                             [63,1],
                             [71,0],
                             [79,1],
                             [85,6],
                             [47,15],
                             [47,23],
                             [47,32],
                             [47,41],
                             [38,46],
                             [43,48],
                             [48,49],
                             [52,48],
                             [57,46],
                             [19,16],
                             [24,13],
                             [30,13],
                             [35,17],
                             [29,18],
                             [23,18],
                             [59,17],
                             [65,12],
                             [71,12],
                             [76,15],
                             [71,17],
                             [65,18],
                             [29,62],
                             [36,59],
                             [43,57],
                             [48,58],
                             [53,57],
                             [60,58],
                             [67,61],
                             [60,68],
                             [54,71],
                             [48,72],
                             [43,72],
                             [36,69],
                             [32,62],
                             [43,62],
                             [48,62],
                             [53,61],
                             [64,61],
                             [53,65],
                             [48,66],
                             [43,65]\
                             ])


    @staticmethod
    def align(im, src_pts):
        """
        given an image and the src points, align it
        """
        im = align_face(im,\
                        src_pts[OpenFaceAlign.selected_indices],\
                        OpenFaceAlign.target_pts[OpenFaceAlign.selected_indices],\
                        OpenFaceAlign.dim)
        return im
