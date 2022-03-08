""" 
Module to calculate the B1 from two datasets with difference in flip angle

    
Erik van der Bijl
    

"""

import numpy as np
import pandas as pd
from dicom_utils import Dicom_Series
import utils


def calc_b1_map(voxel_data_120,voxel_data_60):
    """"
    calculates the b1 map by taking two datasets that have flip angles 120 and 60 degrees.
    """
    # return result of calculation; eps is added for numerical stability
    eps = 1e-6
    return np.arccos(voxel_data_120 / (2. * voxel_data_60+eps))*3./np.pi

def get_B1_map(idx_data, orientation='t',slice_nr=1,radius=175.):
    """
    Returns a pandas dataframe containing information about b1 maps
    
    Parameters:
    -----------
    idx_data : pd.Dataframe
        Pandas Dataframe containing SeriesDescription, AcquisitionDate and fn
    
    orientation : string default 't' 
        String 't', 's' or 'c'
        
    slice_nr: int default 1 
        Slice nr of image to calculate B1 map for.
        
    radius: float default: 175.0
        Radius of the circular mask 
        
    Returns:
    --------
    Pandas Dataframe with date, type, units, orientation, slice_nr,r,institution,machine and data:np.array of 
    
    """    
    # Get all transverse B1 series
    df_B1 = idx_data.loc[idx_data['SeriesDescription'].str.contains(orientation+' B1_')]
       
    # Create list that contains results
    B1_data = []
    
    # loop over all acquisition dates in the index
    for date,B1_study in df_B1.groupby('AcquisitionDate'):
        try:
            #Open dicom Series
            B1_60 = Dicom_Series(B1_study.loc[B1_study['SeriesDescription'].str.contains('60')].fn.values)
            B1_120 = Dicom_Series(B1_study.loc[B1_study['SeriesDescription'].str.contains('120')].fn.values)
        
            #Calculate B1_map
            B1_map = calc_b1_map(B1_120.voxel_data,B1_60.voxel_data)
    
            #select central slice
            B1_map=B1_map[slice_nr]
            
            #Create and apply circular mask
            mask = utils.create_circular_mask(B1_map.shape,B1_60.voxel_spacing[1:],radius=radius)
            
            B1_map[~mask]=np.nan
            
            dset = {
                'date':date,
                'type' :'B1',
                'units': 'relative',
                 'orientation':orientation,
                 'slice_nr':slice_nr,
                 'r':radius,
                 'institution':B1_60.header.InstitutionName,
                 'machine':B1_60.header.StationName,
                 'data': B1_map}
                    
            B1_data.append(dset) 
            

        except Exception as err:
            print(err)
            print(orientation)
            print("Date", date)
#             print(B1_study)
              
    df_B1 = pd.DataFrame(B1_data)
    
    return df_B1