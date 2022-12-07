#############################
# Various Graph Cut Algorithms which operate on the ImageGraph 
# representation of an image. Seeding of any sort is represented in the 
# ImageGraph object. These algorithms should further be able to label 
# the non-seeded pixels in the ImageGraph object they are given. 
#
# Matthew Hokinson, 12/7/22
#############################

from __future__ import annotations
from imageGraph import ImageGraph 

def pushRelabelCut(imageGraph: ImageGraph):
    """
    Push-Relabel Graph Cut Algorithm

    Args:
        imageGraph (ImageGraph): ImageGraph representation of the image
    """
    raise NotImplementedError