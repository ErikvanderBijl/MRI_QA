

import pydicom as pydicom
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from multiprocessing import Pool
from functools import partial

def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    try:
        orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
        position = tuple((float(p) for p in dcm.ImagePositionPatient))
        rowvec, colvec = orientation[:3], orientation[3:]
        normal_vector = np.cross(rowvec, colvec)
        slice_pos = np.dot(position, normal_vector)
    except:
        print('Error: ',dcm.ImageType)
    return slice_pos

class Dicom_Series():
	""""
	Class that loads dicomseries as a whole. 
	
	:method: load_data(dcm_filenames) loads all dicom files in the list
	
	
	"""
    def __init__(self,dcm_filenames):
        self.load_data(dcm_filenames)

    def _get_cube_orientation(self, image_orientation):
        u_ori, v_ori = np.split(np.array(image_orientation, dtype=float), 2)
        plane_normal = np.cross(u_ori, v_ori)
        return np.array([plane_normal, u_ori, v_ori])

    def load_data(self,dcm_filenames):

        # ensure that AcquisitionDate/Time is also set in this function:
        dcm_slices = []
        for fn in dcm_filenames:
        	dcm_temp = pydicom.read_file(fn)
			# fix to avoid problems with missing acquisitionDates
        	dcm_temp.AcquisitionDate = dcm_temp.ContentDate
        	dcm_temp.AcquisitionTime = dcm_temp.ContentTime
			# append to list
        	dcm_slices += [dcm_temp]
        
        #check single series
        series_in_list = set([dcm.SeriesNumber for dcm in dcm_slices])
        if len(series_in_list)>1:
            print('Multiple series in dataset',series_in_list)
            
        #load data
        self.header = dcm_slices[0]
        self.slices = dcm_slices
        
        # Extract position for each slice to sort and calculate slice spacing
        dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcm_slices]
        dcm_slices = sorted(dcm_slices, key=itemgetter(1))
    
        spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
        slice_spacing = np.mean(spacings)

        # All slices will have the same in-plane shape
        shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
  
        self.nslices = len(dcm_slices)

        # Final 3D array will be N_Slices x Columns x Rows
        self.shape = (self.nslices, shape[1],shape[0])
        self.voxel_data = np.empty(self.shape, dtype='float32')
        slope = 1
        intercept = 0
        for idx, (dcm, _) in enumerate(dcm_slices):
            # Rescale and shift in order to get accurate pixel values
            try:

                slope = float(dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope)
                intercept = float(dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept)

            except:
                slope = float(dcm.RescaleSlope)
                intercept = float(dcm.RescaleIntercept)
            self.voxel_data[idx,:,:] = dcm.pixel_array.astype('float32') * slope + intercept
        self.slope = slope
        self.intercept = intercept
        # Calculate size of a voxel in mm
        pixel_spacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)

        self.voxel_spacing = (slice_spacing, *pixel_spacing)
        self.origin = np.array(dcm_slices[0][0].ImagePositionPatient).astype('float32')

        self.axs = self._get_cube_orientation(self.header.ImageOrientationPatient)


def get_dcm_info(fn,extra_tags = ['EchoTime']):
	# AW: replaced AcquisitionDate/Time with ContentDateTime
    tags =  ['SeriesDescription','SeriesInstanceUID','StudyInstanceUID','ContentDate','ContentTime','InstitutionName','InstanceNumber','TemporalPositionIdentifier','ImageType','SeriesNumber']
    if len(extra_tags)>0:
        tags.extend(extra_tags)
    result = {}
    print(tags)
    try:
        dcm = pydicom.read_file(fn,force=True,specific_tags=tags)

        result ={tag:dcm[tag].value for tag in tags}
        # AW: make sure AcquisitionDate/Time is always set:
        result.update({'AcquisitionDate': dcm['ContentDate'].value, 'AcquisitionTime': dcm['ContentTime'].value})
        result.update({'fn':fn})
    except:
        pass
    result.update({'fn':fn})    
    return result

def index_dicom_files(root_directory,extra_tags=[]):
    fn_list = []
    for root, dirs, files in os.walk(root_directory,topdown=False):
        if files:
            fn_list.extend([os.path.join(root,fn) for fn in files])
    p = Pool(4)

    df = pd.DataFrame(p.map(partial(get_dcm_info, extra_tags=extra_tags),fn_list))
    
    try:
        df['ImageType']=df['ImageType'].astype('str')
    except:
        pass
    
    # drop duplicates if same dicom is saved on multiple locations
    df.drop_duplicates(subset=df.columns.difference(['fn']),inplace=True,ignore_index=True)
    
    # Sort dataframe
    df.sort_values(by = ['AcquisitionDate','SeriesInstanceUID','TemporalPositionIdentifier','InstanceNumber'],inplace=True)

    return df





