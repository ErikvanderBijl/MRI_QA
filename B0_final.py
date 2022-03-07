import numpy as np
import pandas as pd
from dicom_utils import Dicom_Series
import utils
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt

def ppm_from_philips(dcm_b0map_philips):
    #MHztoppm factors suppressed
    HzToPPM = 1.0/(63.88177)
    
    return dcm_b0map_philips.voxel_data*HzToPPM
	
def Analyse_B0(idx_data,orientation='t ',slice_nr=1,radius=175):
    
    
    #Select all B0 series and code for dual echo or B0map by philips
    df_B0_normal = idx_data.loc[ idx_data['SeriesDescription'].str.contains(orientation+' 2DFFE_B0')]
    df_B0_special = idx_data.loc[ idx_data['SeriesDescription'].str.contains(orientation+' B0map')]
    df_B0 = pd.concat([df_B0_normal, df_B0_special])
    df_B0['B0_type'] = np.where(df_B0.SeriesDescription.str.contains('echo'),'B0_Dual_echo','B0_Philips')
	
	# Select only Philips B0map for analysis
    df_B0 = df_B0[df_B0.B0_type=='B0_Philips']
    B0_data=[]
    for date,B0_study in df_B0.groupby(['AcquisitionDate']):
       
        date
        try:
            #load dicom series from files
			dcm_B0 = Dicom_Series(B0_study.loc[ B0_study['ImageType'].str.contains('B0 MAP')].fn.values)    
            
			#calculate B0 map
			B0_map = ppm_from_philips(dcm_B0)
			
			#apply circular mask to B0map
            mask = utils.create_circular_mask(dcm_B0.shape[1:],dcm_B0.voxel_spacing[1:],radius=radius)        
            B0_map[0,~mask]=np.nan
            B0_map[1,~mask]=np.nan
            B0_map[2,~mask]=np.nan
            B0_map[3,~mask]=np.nan
            B0_map[4,~mask]=np.nan
                
            dset = {'date':date,
                    'type' :B0_type,
                    'units': 'ppm',
                    'orientation':orientation,
                    'r':radius,
                    'institution':dcm_B0.header.InstitutionName,
                    'machine':dcm_B0.header.StationName,
                    'TE': dcm_B0.header.EchoTime,
                    'data': np.nanmean(B0_map[0:5],axis=0) #mean data over 5 slices
					 }

            B0_data.append(dset) 
        except Exception as err:
            print(err)
            print("Date", date)

    return pd.DataFrame(B0_data)