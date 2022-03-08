""" 
Module to calculate GB0.

    
Erik van der Bijl
    

"""

import numpy as np
import pandas as pd
from dicom_utils import Dicom_Series
import utils
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt

def Analyse_gantry_B0(idx_data,slice_nr = 2,radius=175):
    # list of acquired gantry positions
    gantry_angles=[-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    
    # Selection of datasets from index
    df_GantryB0 = idx_data.loc[idx_data['SeriesDescription'].str.contains('SHIM') &
                               idx_data['ImageType'].str.contains('\'P\'')]
   
    #list to store results
    GB0_data = []
    
    #analyze per acquisition date
    for date, GB0 in df_GantryB0.groupby('AcquisitionDate'):
        
        try:
            print("Date", date)
            # Load dynamic dicom series
            dcmGB0 = Dicom_Series(GB0.fn.values)
        
            # Reshape into gantry angle x num images x shape
            phase_map_Gs = np.reshape(dcmGB0.voxel_data,(-1,3,256,256),'F')
        
       
         
            #mask for removing exterior     
            mask = utils.create_circular_mask((256,256),dcmGB0.voxel_spacing[1:],radius=radius)
                              
            #Check correct number of datasets is found
            if(phase_map_Gs.shape[0]!=13):
                print(date,'Strange number of datasets')
                print(phase_map_Gs.shape)
                continue
            # Select gantry 0 as reference
            phase_map_G0 = phase_map_Gs[6]
           
            for phase_map_Ga,angle in zip(phase_map_Gs,gantry_angles):
            
#                unwrap phase division by 1000 to bring phase (-pi,pi)
            
                phase_Ga_unwrapped = unwrap_phase(phase_map_Ga[slice_nr]/1000.)
                phase_Gref_unwrapped =unwrap_phase(phase_map_G0[slice_nr]/1000.)
        
                phase_diff=(phase_Ga_unwrapped-phase_Gref_unwrapped)
            
                #correct centre of unwrapping for factors of +/- 2pi
                if (abs(phase_diff[128,128])>np.pi):
                    phase_diff-=2*np.pi*np.sign(phase_diff[128,128])
           
                #apply 
                phase_diff[~mask]=np.nan


                ppm = phase_diff/(2*np.pi*15.66*1e-3*63.88177)
            
                dset = {'date':date,
                    'type' :'GB0',
                    'units': 'nT',
                    'Gantry':angle,
                    'Orientation':'t',
                    'Slice':slice_nr,
                    'R':radius,
                    'Institution':dcmGB0.header.InstitutionName,
                    'Machine':dcmGB0.header.StationName,
                    'data': ppm}
                GB0_data.append(dset)
                
        except:
            print(err)
            #print("Date", date)
        
    return pd.DataFrame(GB0_data)
