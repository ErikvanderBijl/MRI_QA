#settings for MRL
import numpy as np
TRANSVERSEORIENTATION = [1, 0, 0, 0, 1, 0]

CLUSTERSIZE = 500
LIMIT_Z_SEPARATION_FROM_TABLEPOSITION=27

#Labels of directions
AP=1
LR=0
CC=2

LIMITFITDEGREES = 4 * 3.14159265358979 / 180 #radians -> 4 degrees
LIMITFITTRANS = 5 #mm

DIST_TO_ISOC_RIGID = 5  # Take into account the (5x5+1)^2 closest positions to isoc
CLUSTER_SEPARATION = 25

LIMITMAXDISTANCE=2

LIMIT_CC_SEPARATION_FROM_CC_POSITION = 25.0
marker_positions_CC = [-165.0,-110.0,-55.0,0.0,55.0,110.0,165.0]

markerPositions_LR_AP=np.array([
                       (-250.000,-50.000),
                       (-250.000, -25.000),
                       (-250.000,  +0.000),
                       (-250.000, +25.000),
                       (-250.000, +50.000),
                       (-225.000,-100.000),
                       (-225.000, -75.000),
                       (-225.000, -50.000),
                       (-225.000, -25.000),
                       (-225.000,  +0.000),
                       (-225.000, +25.000),
                       (-225.000, +50.000),
                       (-225.000, +75.000),
                       (-225.000,+100.000),
                       (-225.000,+125.000),
                       (-200.000,-150.000),
                       (-200.000,-125.000),
                       (-200.000,-100.000),
                       (-200.000, -75.000),
                       (-200.000, -50.000),
                       (-200.000, -25.000),
                       (-200.000,  +0.000),
                       (-200.000, +25.000),
                       (-200.000, +50.000),
                       (-200.000, +75.000),
                       (-200.000,+100.000),
                       (-200.000,+125.000),
                       (-175.000,-175.000),
                       (-175.000,-150.000),
                       (-175.000,-125.000),
                       (-175.000,-100.000),
                       (-175.000, -75.000),
                       (-175.000, -50.000),
                       (-175.000, -25.000),
                       (-175.000,  +0.000),
                       (-175.000, +25.000),
                       (-175.000, +50.000),
                       (-175.000, +75.000),
                       (-175.000,+100.000),
                       (-175.000,+125.000),
                       (-150.000,-200.000),
                       (-150.000,-175.000),
                       (-150.000,-150.000),
                       (-150.000,-125.000),
                       (-150.000,-100.000),
                       (-150.000, -75.000),
                       (-150.000, -50.000),
                       (-150.000, -25.000),
                       (-150.000,  +0.000),
                       (-150.000, +25.000),
                       (-150.000, +50.000),
                       (-150.000, +75.000),
                       (-150.000,+100.000),
                       (-150.000,+125.000),
                       (-125.000,-200.000),
                       (-125.000,-175.000),
                       (-125.000,-150.000),
                       (-125.000,-125.000),
                       (-125.000,-100.000),
                       (-125.000, -75.000),
                       (-125.000, -50.000),
                       (-125.000, -25.000),
                       (-125.000,  +0.000),
                       (-125.000, +25.000),
                       (-125.000, +50.000),
                       (-125.000, +75.000),
                       (-125.000,+100.000),
                       (-125.000,+125.000),
                       (-100.000,-225.000),
                       (-100.000,-200.000),
                       (-100.000,-175.000),
                       (-100.000,-150.000),
                       (-100.000,-125.000),
                       (-100.000,-100.000),
                       (-100.000, -75.000),
                       (-100.000, -50.000),
                       (-100.000, -25.000),
                       (-100.000,  +0.000),
                       (-100.000, +25.000),
                       (-100.000, +50.000),
                       (-100.000, +75.000),
                       (-100.000,+100.000),
                       (-100.000,+125.000),
                       ( -75.000,-225.000),
                       ( -75.000,-200.000),
                       ( -75.000,-175.000),
                       ( -75.000,-150.000),
                       ( -75.000,-125.000),
                       ( -75.000,-100.000),
                       ( -75.000, -75.000),
                       ( -75.000, -50.000),
                       ( -75.000, -25.000),
                       ( -75.000,  +0.000),
                       ( -75.000, +25.000),
                       ( -75.000, +50.000),
                       ( -75.000, +75.000),
                       ( -75.000,+100.000),
                       ( -75.000,+125.000),
                       ( -50.000,-250.000),
                       ( -50.000,-225.000),
                       ( -50.000,-200.000),
                       ( -50.000,-175.000),
                       ( -50.000,-150.000),
                       ( -50.000,-125.000),
                       ( -50.000,-100.000),
                       ( -50.000, -75.000),
                       ( -50.000, -50.000),
                       ( -50.000, -25.000),
                       ( -50.000,  +0.000),
                       ( -50.000, +25.000),
                       ( -50.000, +50.000),
                       ( -50.000, +75.000),
                       ( -50.000,+100.000),
                       ( -50.000,+125.000),
                       ( -25.000,-250.000),
                       ( -25.000,-225.000),
                       ( -25.000,-200.000),
                       ( -25.000,-175.000),
                       ( -25.000,-150.000),
                       ( -25.000,-125.000),
                       ( -25.000,-100.000),
                       ( -25.000, -75.000),
                       ( -25.000, -50.000),
                       ( -25.000, -25.000),
                       ( -25.000,  +0.000),
                       ( -25.000, +25.000),
                       ( -25.000, +50.000),
                       ( -25.000, +75.000),
                       ( -25.000,+100.000),
                       ( -25.000,+125.000),
                       (  +0.000,-250.000),
                       (  +0.000,-225.000),
                       (  +0.000,-200.000),
                       (  +0.000,-175.000),
                       (  +0.000,-150.000),
                       (  +0.000,-125.000),
                       (  +0.000,-100.000),
                       (  +0.000, -75.000),
                       (  +0.000, -50.000),
                       (  +0.000, -25.000),
                       (  +0.000,  +0.000),
                       (  +0.000, +25.000),
                       (  +0.000, +50.000),
                       (  +0.000, +75.000),
                       (  +0.000,+100.000),
                       (  +0.000,+125.000),
                       ( +25.000,-250.000),
                       ( +25.000,-225.000),
                       ( +25.000,-200.000),
                       ( +25.000,-175.000),
                       ( +25.000,-150.000),
                       ( +25.000,-125.000),
                       ( +25.000,-100.000),
                       ( +25.000, -75.000),
                       ( +25.000, -50.000),
                       ( +25.000, -25.000),
                       ( +25.000,  +0.000),
                       ( +25.000, +25.000),
                       ( +25.000, +50.000),
                       ( +25.000, +75.000),
                       ( +25.000,+100.000),
                       ( +25.000,+125.000),
                       ( +50.000,-250.000),
                       ( +50.000,-225.000),
                       ( +50.000,-200.000),
                       ( +50.000,-175.000),
                       ( +50.000,-150.000),
                       ( +50.000,-125.000),
                       ( +50.000,-100.000),
                       ( +50.000, -75.000),
                       ( +50.000, -50.000),
                       ( +50.000, -25.000),
                       ( +50.000,  +0.000),
                       ( +50.000, +25.000),
                       ( +50.000, +50.000),
                       ( +50.000, +75.000),
                       ( +50.000,+100.000),
                       ( +50.000,+125.000),
                       ( +75.000,-225.000),
                       ( +75.000,-200.000),
                       ( +75.000,-175.000),
                       ( +75.000,-150.000),
                       ( +75.000,-125.000),
                       ( +75.000,-100.000),
                       ( +75.000, -75.000),
                       ( +75.000, -50.000),
                       ( +75.000, -25.000),
                       ( +75.000,  +0.000),
                       ( +75.000, +25.000),
                       ( +75.000, +50.000),
                       ( +75.000, +75.000),
                       ( +75.000,+100.000),
                       ( +75.000,+125.000),
                       (+100.000,-225.000),
                       (+100.000,-200.000),
                       (+100.000,-175.000),
                       (+100.000,-150.000),
                       (+100.000,-125.000),
                       (+100.000,-100.000),
                       (+100.000, -75.000),
                       (+100.000, -50.000),
                       (+100.000, -25.000),
                       (+100.000,  +0.000),
                       (+100.000, +25.000),
                       (+100.000, +50.000),
                       (+100.000, +75.000),
                       (+100.000,+100.000),
                       (+100.000,+125.000),
                       (+125.000,-200.000),
                       (+125.000,-175.000),
                       (+125.000,-150.000),
                       (+125.000,-125.000),
                       (+125.000,-100.000),
                       (+125.000, -75.000),
                       (+125.000, -50.000),
                       (+125.000, -25.000),
                       (+125.000,  +0.000),
                       (+125.000, +25.000),
                       (+125.000, +50.000),
                       (+125.000, +75.000),
                       (+125.000,+100.000),
                       (+125.000,+125.000),
                       (+150.000,-200.000),
                       (+150.000,-175.000),
                       (+150.000,-150.000),
                       (+150.000,-125.000),
                       (+150.000,-100.000),
                       (+150.000, -75.000),
                       (+150.000, -50.000),
                       (+150.000, -25.000),
                       (+150.000,  +0.000),
                       (+150.000, +25.000),
                       (+150.000, +50.000),
                       (+150.000, +75.000),
                       (+150.000,+100.000),
                       (+150.000,+125.000),
                       (+175.000,-175.000),
                       (+175.000,-150.000),
                       (+175.000,-125.000),
                       (+175.000,-100.000),
                       (+175.000, -75.000),
                       (+175.000, -50.000),
                       (+175.000, -25.000),
                       (+175.000,  +0.000),
                       (+175.000, +25.000),
                       (+175.000, +50.000),
                       (+175.000, +75.000),
                       (+175.000,+100.000),
                       (+175.000,+125.000),
                       (+200.000,-150.000),
                       (+200.000,-125.000),
                       (+200.000,-100.000),
                       (+200.000, -75.000),
                       (+200.000, -50.000),
                       (+200.000, -25.000),
                       (+200.000,  +0.000),
                       (+200.000, +25.000),
                       (+200.000, +50.000),
                       (+200.000, +75.000),
                       (+200.000,+100.000),
                       (+200.000,+125.000),
                       (+225.000,-100.000),
                       (+225.000, -75.000),
                       (+225.000, -50.000),
                       (+225.000, -25.000),
                       (+225.000,  +0.000),
                       (+225.000, +25.000),
                       (+225.000, +50.000),
                       (+225.000, +75.000),
                       (+225.000,+100.000),
                       (+225.000,+125.000),
                       (+250.000, -50.000),
                       (+250.000, -25.000),
                       (+250.000,  +0.000),
                       (+250.000, +25.000),
                       (+250.000, +50.000)])


