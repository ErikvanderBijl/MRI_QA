import os
import pandas as pd
import re
import numpy as np

def get_piqt_files(root_dir):
    # simple function that returns returns a list of all absolute paths in .htm and .html files in the given folder and it subfolders
    return [os.path.join(root,file)for root, dirs, files in os.walk(root_dir)for file in files if (file.endswith('.htm') or file.endswith('.html')  )]
            
def get_piqtdata(html_files,encoding = 'utf8'):
    """ reads PIQT html or htm reports to pandas DataFrame
    
    
    """
            
    # List of dataframes to be concatenated at the end            
    dfs = []
            
    # Loop over html_files and open them
    for html_piqt in html_files:
        with open(html_piqt,'r',encoding=encoding) as fp:     

            # Split into parts using regular exprions mating
            for part in (re.split(r'<span>',fp.read()))[1:]:
                try:
                    # Extract measurement name
                    measurement = (re.findall('>(.*?)<',part)[2])

                    # Read all tables
                    tbls = pd.read_html(part)            

                    # Split data in second table into single column
                    tbls[1]=(pd.concat([tbls[1][[0,1]],tbls[1][[2,3]].rename(columns={2:0,3:1})])).dropna()

                    # odd rows contain limits in third table
                    df_odd = tbls[2].iloc[:,1::2].copy()
                    df_even = tbls[2].iloc[:,2::2]

                    # detect rows containing limits
                    df_even.columns=df_odd.columns
                    idx =(df_odd!=df_even).all(axis=1)

                    # Combine limits and values into new table
                    limits = (tbls[2][idx].iloc[:,0]+'_limit')
                    df_append = df_even[idx].copy()
                    df_append[0]=limits
                    df_odd[0] = tbls[2].iloc[:,0]
                    tbls[2] = (pd.concat([df_odd,df_append]))


                    # Concat dataframes
                    df=pd.concat([tbl.set_index(tbl[0]) for tbl in tbls])

                    # Fill n/a values 
                    df.fillna(method="ffill" , inplace=True, axis=1)

                    # Drop 
                    df = df.drop(columns=0)


                    #transpose: rows are measurements now
                    df = df.T

                    #add label for PIQT measurement series
                    df['measurement'] = measurement

                    dfs.append(df)
                except Exception as e:
                    print(e)
                    pass
                
    #concatenate all dataframes
    df_piqt = pd.concat(dfs,ignore_index=True)
    
    #Convert selected columns to datetime            
    df_piqt['Date'] = pd.to_datetime(df_piqt['Date'])
    df_piqt['Scan_Date'] = pd.to_datetime(df_piqt['Scan_Date'],format='%d-%m-%Y')
    
    # set N/A to numpy Nan to be able to convert to floats where the column only contains values that can be converted to float.
    df_piqt.replace('NaN', np.nan)
    for c in df_piqt.columns:
        try:
            if df_piqt[c].dtype=='object':
                
                df_piqt[c] = df_piqt[c].astype(float)
        except:
            pass
    
    return df_piqt
