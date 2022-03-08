import os
import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
import scipy
import logging
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from datetime import datetime,date,time

from scipy.cluster.vq import kmeans2
from multiprocessing import Pool

#initiate logger
logger = logging.getLogger(__name__)
handler = logging.FileHandler('GeomAcc')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def _parallel_cluster(points):


    result = []
    cur_points=np.copy(points)
    while len(cur_points) > 0:
        curPoint = cur_points[0]
        squaredDistanceToCurrentPoint = cdist([curPoint], cur_points, 'sqeuclidean')[0]
        pointsCloseToCurrent = squaredDistanceToCurrentPoint < 225
        result.append(np.mean(cur_points[pointsCloseToCurrent], axis=0))
        cur_points = cur_points[np.logical_not(pointsCloseToCurrent)]
    return np.array(result)


class GeomAcc():

    def __init__(self,module="MRL"):

        import GeomAccDefaultsMRL as GeomAccDefaults

        #Define default constants
        self.LR = GeomAccDefaults.LR
        self.AP = GeomAccDefaults.AP
        self.CC = GeomAccDefaults.CC

        #Properties of the study
        self.studyDate = None
        self.studyTime = None
        self.studyScanner = None

        #results for this study
        self.rigid_transformation_setup = [0, 0, 0, 0, 0, 0]
        self.measurementsPerTablePos = {}
        self.measurementTablePositions = []

        #Constants/Config
        self.TRANSVERSE_ORIENTATION = GeomAccDefaults.TRANSVERSEORIENTATION
        self.MARKER_THRESHOLD_AFTER_FILTERING = GeomAccDefaults.MARKER_THRESHOLD_AFTER_FILTERING
        self.CLUSTERSIZE = GeomAccDefaults.CLUSTERSIZE
        self.LIMIT_CC_SEPARATION_FROM_CC_POSITION=GeomAccDefaults.LIMIT_CC_SEPARATION_FROM_CC_POSITION

        self.positions_CC = np.sort(np.array(GeomAccDefaults.marker_positions_CC))
        self.positions_LR_AP = np.array(GeomAccDefaults.markerPositions_LR_AP,dtype=float)

        self.degLimit = GeomAccDefaults.LIMITFITDEGREES
        self.transLimit = GeomAccDefaults.LIMITFITTRANS

        self.expectedMarkerPositions = self._expected_marker_positions(self.positions_CC, self.positions_LR_AP)

        #Results
        self.detectedMarkerPositions  = None
        self.correctedMarkerPositions = None
        self.closestExpectedMarkerIndices = None
        self.differencesCorrectedExpected = None

    def _expected_marker_positions(self, marker_positions_cc, marker_positions_LR_AP):
        #create a complete list of marker positions by copying the AP_LR
        # list to every cc position that is scanned
        expected_marker_positions= np.vstack([self.__marker_positions_at_cc_pos(cc_pos,marker_positions_LR_AP) for cc_pos in marker_positions_cc])
        return expected_marker_positions

    def __marker_positions_at_cc_pos(self, cc_pos, marker_positions_LR_AP):
        return np.hstack((marker_positions_LR_AP,
                          np.ones((len(marker_positions_LR_AP), 1)) * cc_pos))

    def loadSeries(self,dcmSeries):
        logger.log(logging.INFO, 'loading data from series')

        #Read Dicom header data
        self.readHeaderData(dcmSeries)

		#get detected points
        detectedPoints=self._getHighIntensityPoints(dcmSeries)

        #Cluster high contrast points into markerpositions
        self.detectedMarkerPositions=self._createClusters(detectedPoints,size=self.CLUSTERSIZE)
        logger.log(logging.INFO,self.detectedMarkerPositions)

    def readHeaderData(self,dcmSeries):
		#Read some header data
        self.studyScanner=dcmSeries.header.PerformedStationAETitle
        self.seriesDescription = dcmSeries.header.SeriesDescription

    def _getHighIntensityPoints(self, dcmSeries):
		# detects high intensity points from dcmSeries
        self.origin = dcmSeries.origin
        self.spacing = dcmSeries.voxel_spacing
        self.axs = dcmSeries.axs

        highVoxels = self._getHighVoxelsFromImageData(np.swapaxes(dcmSeries.voxel_data,1,2))
        highPoints = self.index_to_coords(highVoxels)

        return highPoints

    def index_to_coords(self,ix):
        return self.origin+np.dot((ix.T*self.spacing),self.axs)

    def _getHighVoxelsFromImageData(self, imageData):
	    # filter with gaussian kernel and get highest intensity voxels.
        logger.log(logging.INFO, "Filtering dataset")

        sigma = 2.5
        dataCube = scipy.ndimage.filters.gaussian_filter(imageData, sigma, truncate=5)/ np.sqrt(2*np.pi * sigma**2)
        idx = np.argwhere(dataCube > np.percentile(dataCube,99.5)).T
        return idx

    def _createClusters(self, points, size):
        logger.log(logging.INFO, "Finding clusters of high intensity voxels")

        #k-means clustering to separate distinct marker planes
        # Cluster according to expected CC positions of phantomslabs
        centroids,cluster_id= kmeans2(points.T[self.CC], self.positions_CC)
        

        points_per_cc = [points[cluster_id == n_cluster] for n_cluster in np.arange(len(self.positions_CC))]
        
        workpool = Pool(6)
        clusters = np.concatenate(workpool.map(_parallel_cluster,points_per_cc[:]))
                
        return clusters

    def indices_cc_pos(self, positions, cc_pos):
        return np.abs(positions.T[self.CC] - cc_pos) < self.LIMIT_CC_SEPARATION_FROM_CC_POSITION

    def setCorrectedMarkerPositions(self,transformation):
        self.correctedMarkerPositions = self.rigidTransform(self.detectedMarkerPositions,transformation[0:3],[transformation[3], transformation[4], transformation[5]])
        self.closestExpectedMarkerIndices = self.closestLocations(self.correctedMarkerPositions,self.expectedMarkerPositions)
        self.differencesCorrectedExpected = self.getdifferences(self.correctedMarkerPositions,self.expectedMarkerPositions)

    def closestLocations(self, detectedMarkerPositions, expectedMarkerPositions):
        distances = np.sum(np.power((detectedMarkerPositions - expectedMarkerPositions[:, np.newaxis]), 2), axis=2)
        return np.argmin(distances, axis=0)

    def getdifferences(self, markerPositions,expectedMarkerPositions):
        return markerPositions - expectedMarkerPositions[self.closestLocations(markerPositions,expectedMarkerPositions)]

    def _findMarkerIndex(self, xyz):
        x,y,z=zip*(xyz)
        xPosMarkers, yPosMarkers,zPosMarkers = self.expectedMarkerPositions.T
        xMatch = (x == xPosMarkers)
        yMatch = (y == yPosMarkers)
        zMatch = (z == zPosMarkers)
        return np.argwhere(np.logical_and(np.logical_and(xMatch, yMatch),zMatch))[0]


    def correctDetectedClusterPositionsForSetup(self):
        """
        This function adds corrected clusterpositions to the measurements based on the
        calculated setup rotation and translation in the measurement at tableposition 0
        :return:
        """
        logger.log(logging.INFO, "Correcting for phantom setup")

        #Select only markers within 80 mm of isoc
        detected_markers_at_isoc_plane = self.detectedMarkerPositions[self.indices_cc_pos(self.detectedMarkerPositions,cc_pos=0.0)]
        dist_to_isoc_2d = cdist([[0.0, 0.0]], detected_markers_at_isoc_plane[:,:-1], metric='euclidean')[0]
        ix=dist_to_isoc_2d<100

        detected_markers_at_isoc_plane=detected_markers_at_isoc_plane[ix]
        markers_at_isoc_plane = self.__marker_positions_at_cc_pos(0.0, self.positions_LR_AP)
        
#         print(detected_markers_at_isoc_plane)
        

        self.rigid_transformation_setup = self._findRigidTransformation(detected_markers_at_isoc_plane,markers_at_isoc_plane)
        logger.log(logging.INFO,self.rigid_transformation_setup)
        self.setCorrectedMarkerPositions(self.rigid_transformation_setup)

    def _findRigidTransformation(self, detectedMarkerPositions, expectedMarkerPositions):
        logger.log(logging.INFO, "Determining setup translation and rotation for tableposition 0")

        # average detected cc position
        init_CC = np.mean(detectedMarkerPositions, axis=0)[self.CC]

        # optimization init
        optimization_initial_guess = np.zeros(6)
        optimization_initial_guess[self.CC] = init_CC

        # optimization bounds
        optimization_bounds = [(-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit)]

        def penaltyFunction(transRot):
            opt_pos = self.rigidTransform(detectedMarkerPositions, translation=transRot[0:3], eulerAngles=transRot[3:6])
            differences = self.getdifferences(opt_pos, expectedMarkerPositions)
            penalty = np.sum(np.power(differences, 2))
            return penalty

        opt_result = optimize.minimize(fun=penaltyFunction,
                                       x0=optimization_initial_guess,
                                       bounds=optimization_bounds,
                                       tol=.00001)

        return opt_result.x

    def rigidRotation(self, markerPositions, eulerAngles):
        s0 = np.sin(eulerAngles[0])
        c0 = np.cos(eulerAngles[0])
        s1 = np.sin(eulerAngles[1])
        c1 = np.cos(eulerAngles[1])
        s2 = np.sin(eulerAngles[2])
        c2 = np.cos(eulerAngles[2])

        m00 = c1 * c2
        m01 = c0 * s2 + s0 * s1 * c2
        m02 = s0 * s2 - c0 * s1 * c2
        m10 = -c1 * s2
        m11 = c0 * c2 - s0 * s1 * s2
        m12 = s0 * c2 + c0 * s1 * s2
        m20 = s1
        m21 = -s0 * c1
        m22 = c0 * c1

        rotationMatrix = np.array([
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]])

        return np.dot(markerPositions, rotationMatrix)

    def rigidTranslation(self, markerPositions, Translation):
        return markerPositions + Translation

    def rigidTransform(self, positions, translation, eulerAngles):
        rotated_positions = self.rigidRotation(positions, eulerAngles)
        transformed_positions = self.rigidTranslation(rotated_positions, translation)
        return transformed_positions