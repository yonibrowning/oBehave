import numpy as np
import pandas as pd
from datetime import datetime

def includeDateTime(manifest,dateStringKey = 'experiment_date',datetimeKey = 'experiment_datetime',dateConversionStr = '%m/%d/%Y %H:%S'):
    experiment_date = manifest[dateStringKey].values
    experiment_datetime = []
    for date in experiment_date:
        experiment_datetime.append(datetime.strptime(date,dateConversionStr))
    manifest[datetimeKey] = pd.Series(experiment_datetime)
    return manifest

def includeNovelSession(manifest):
    '''
    Appends a Series to manifest dataframe to indicate if this session is the first time that this mouse did each behavioral task.
    Inputs: Allen style manifest 
    '''
    if not ('experiment_datetime' in manifest.columns):
        includeDateTime(manifest);
    
    unique_donors = np.unique(manifest['donor_id'])
    manifest.sort_values('experiment_datetime',ascending = True)
    manifest['first_session'] = pd.Series(np.zeros(manifest['session_type'].shape,dtype = bool))
    behavior_types = np.unique(manifest.session_type)
    for ii,donor in enumerate(unique_donors):
        thismouse = manifest[manifest.donor_id==donor].sort_values('experiment_datetime',ascending = True)
        for beh in behavior_types:
            if beh in thismouse.session_type.values:
                experiment_id = thismouse[thismouse.session_type==beh]['experiment_id'].values[0]
                Rindex = manifest[manifest['experiment_id']==experiment_id].index[0]
                manifest.at[Rindex,'first_session'] = 1
    return manifest


def moving_response_average(dataset,window = 5, tme = None):
    '''
    Computes a moving average behavior hit rate and correct reject rate.
    Returns hit rate and correct r
    Inputs:
    dataset: behavior dataset from allen tools
    window: moving trial average window (optional, default is 5)
    tme: time steps for upsampling. If None (default) uses dataset.timestamps_ophys
    Returns:
    cont_go_ave: moving average hit rate
    cont_catch_ave: moving average of correct response rate
    '''
    trldata = dataset.trials
    catch_tmes = trldata[trldata.trial_type == 'catch']['change_time']
    go_tmes = trldata[trldata.trial_type=='go']['change_time']
    catch_correct = trldata[trldata.trial_type=='catch']['response_type']=='CR'
    go_correct = trldata[trldata.trial_type=='go']['response_type']=='HIT'

    go_ave = np.convolve(go_correct, np.ones((window,))/window, mode='same')
    catch_ave = np.convolve(catch_correct,np.ones((window,))/window, mode='same')
    
    if tme is None:
        cont_go_ave = np.interp(dataset.timestamps_ophys,go_tmes,go_ave)
        cont_catch_ave = np.interp(dataset.timestamps_ophys,catch_tmes,catch_ave)
    else:
        cont_go_ave = np.interp(tme,go_tmes,go_ave)
        cont_catch_ave = np.interp(tme,catch_tmes,catch_ave)

    return cont_go_ave,cont_catch_ave