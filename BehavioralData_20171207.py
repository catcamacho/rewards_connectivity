
# coding: utf-8

# In[11]:

def timing_bars(run_timing_list,motion, motion_thresh, BOLD_window, subjid, timepoint, behavior_dir):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from pandas import DataFrame,Series,read_table,concat
    from os.path import abspath
     
    # Import and organize motion data
    motion_dfs = [ read_table(j,delim_whitespace=True,header=None, names=['motion']) for j in motion ]
    motion_fd = concat(motion_dfs,ignore_index=True)

    # Create full task dataframe
    run_timing_list = sorted(run_timing_list)
    dfs = [ read_table(i,delim_whitespace=True) for i in run_timing_list ]
    k=1
    for df in dfs:
        df.loc[:,'runNum'] = Series(k, index = df.index)
        df.loc[:,'time_hyp'] = (k-1)*453 + df.loc[:,'time_hyp']
        df.loc[:,'trial'] = (k*100) + df.loc[:,'trial']
        k = k+1
    df_full = concat(dfs,ignore_index=True)
    df_full = df_full.sort_values(['runNum','time_hyp'], ascending=[1,1])
    df_full.loc[:,'motion'] = motion_fd
    df_full.loc[:,'subjid'] = Series(subjid,index=df_full.index)
    df_full.loc[:,'timepoint'] = Series(timepoint,index=df_full.index)
    # Sort out trials that are complete
    df_complete = df_full[df_full.loc[:,'catch']==0]
    
    # Add accuracy variable
    df_complete['acc'] = Series(1, index=df_complete.index)
    for index, row in df_complete.iterrows():
        if df_complete.loc[index,'Count'] == 2:
            df_complete.loc[index,'acc'] = 0
    
    # Add additional label to the trials with high motion
    df_complete.loc[:,'mot_cat'] = Series('low',index=df_complete.index)
    df_trials = df_complete[df_complete.loc[:,'stim']=='cue']
    for index, row in df_trials.iterrows():
        hrf_length = index+BOLD_window
        trial_motion = df_full.iloc[index:hrf_length,8]
        excess_vols = (trial_motion >= motion_thresh) + (trial_motion <= (-1*motion_thresh))
        if sum(excess_vols) >= 3:
            df_trials.loc[index,'mot_cat'] = 'high'
    
    df_trials.to_csv(behavior_dir + str(subjid) + '_cleaned_behavioral_data.csv')
    return(df_trials)


# In[12]:

from pandas import DataFrame, Series
import matplotlib.pyplot as plt 
from glob import glob

#analysis_home = '/Users/catcamacho/Box/LNCD_rewards_connectivity'
analysis_home = '/Volumes/Zeus/Cat'
#raw_dir = analysis_home + '/subjs'
raw_dir = '/Volumes/Phillips/bars/APWF_bars/subjs'
behavior_dir = analysis_home + '/proc/behavior/'
preproc_dir = analysis_home + '/proc/preprocessing'

#pull subject info to iter over
subject_info = DataFrame.from_csv(analysis_home + '/misc/subjs.csv')
subjects_list = subject_info['SubjID'].tolist()
timepoints = subject_info['Timepoint'].tolist()

#subjects_list = [10766]
#timepoints = [1]
motion_thresh = 0.9
BOLD_window = 8

column_names = ['num_use_total','lat_total','lat_total_std','num_use_neut','mean_lat_neut',
                'std_lat_neut','num_use_pun','mean_lat_pun','std_lat_pun','mot_all',
                'mot_pun','mot_neut', 'acc_all','acc_pun','acc_neut']
lm_cols = ['lm_' + a for a in column_names]
hm_cols = ['hm_' + a for a in column_names]

summary_col_names = column_names + lm_cols + hm_cols
summary_data = DataFrame()
summary_data.loc[:,'subjid'] = Series(subjects_list,index=None)
summary_data.loc[:,'timepoint'] = Series(timepoints,index=summary_data.index)
summary_data = summary_data.reindex(columns= summary_data.columns.tolist() + summary_col_names)

for subjid in subjects_list:
    sub_index = summary_data[summary_data['subjid']==subjid].index[0]
    timepoint = timepoints[subjects_list.index(subjid)]
    motion = glob(preproc_dir + '/FD_out_metric_values/%d_t%d/*/FD.txt' % (subjid,timepoint))
    run_timing_list = glob(raw_dir + '/%d/%d_*/timing/*score_timing.txt'% (subjid,timepoint))
    
    subject_df = timing_bars(run_timing_list,motion, motion_thresh, BOLD_window, subjid, timepoint, behavior_dir)
    sub_neut = subject_df[subject_df['cond'] == 'neutral']
    sub_pun = subject_df[subject_df['cond'] == 'punish']
    
    summary_data.loc[sub_index,'num_use_total'] = subject_df.shape[0]
    summary_data.loc[sub_index,'lat_total'] = subject_df['lat'].mean()
    summary_data.loc[sub_index,'lat_total_std'] = subject_df['lat'].std()
    summary_data.loc[sub_index,'num_use_neut'] = sub_neut.shape[0]
    summary_data.loc[sub_index,'mean_lat_neut'] = sub_neut['lat'].mean()
    summary_data.loc[sub_index,'std_lat_neut'] = sub_neut['lat'].std()
    summary_data.loc[sub_index,'num_use_pun'] = sub_pun.shape[0]
    summary_data.loc[sub_index,'mean_lat_pun'] = sub_pun['lat'].mean()
    summary_data.loc[sub_index,'std_lat_pun'] = sub_pun['lat'].std()
    summary_data.loc[sub_index,'mot_pun'] = sub_pun['motion'].mean()
    summary_data.loc[sub_index,'mot_neut'] = sub_neut['motion'].mean()
    summary_data.loc[sub_index,'mot_all'] = subject_df['motion'].mean()
    summary_data.loc[sub_index,'acc_all'] = subject_df['acc'].mean()
    summary_data.loc[sub_index,'acc_pun'] = sub_pun['acc'].mean()
    summary_data.loc[sub_index,'acc_neut'] = sub_neut['acc'].mean()
    
    lm_df = subject_df[subject_df['mot_cat'] == 'low']
    sub_neut = lm_df[lm_df['cond'] == 'neutral']
    sub_pun = lm_df[lm_df['cond'] == 'punish']
    
    summary_data.loc[sub_index,'lm_num_use_total'] = lm_df.shape[0]
    summary_data.loc[sub_index,'lm_lat_total'] = subject_df['lat'].mean()
    summary_data.loc[sub_index,'lm_lat_total_std'] = lm_df['lat'].std()
    summary_data.loc[sub_index,'lm_num_use_neut'] = sub_neut.shape[0]
    summary_data.loc[sub_index,'lm_mean_lat_neut'] = sub_neut['lat'].mean()
    summary_data.loc[sub_index,'lm_std_lat_neut'] = sub_neut['lat'].std()
    summary_data.loc[sub_index,'lm_num_use_pun'] = sub_pun.shape[0]
    summary_data.loc[sub_index,'lm_mean_lat_pun'] = sub_pun['lat'].mean()
    summary_data.loc[sub_index,'lm_std_lat_pun'] = sub_pun['lat'].std()
    summary_data.loc[sub_index,'lm_mot_pun'] = sub_pun['motion'].mean()
    summary_data.loc[sub_index,'lm_mot_neut'] = sub_neut['motion'].mean()
    summary_data.loc[sub_index,'lm_mot_all'] = lm_df['motion'].mean()
    summary_data.loc[sub_index,'lm_acc_all'] = lm_df['acc'].mean()
    summary_data.loc[sub_index,'lm_acc_pun'] = sub_pun['acc'].mean()
    summary_data.loc[sub_index,'lm_acc_neut'] = sub_neut['acc'].mean()
    
    hm_df = subject_df[subject_df['mot_cat'] == 'high']
    sub_neut = hm_df[hm_df['cond'] == 'neutral']
    sub_pun = hm_df[hm_df['cond'] == 'punish']
    
    summary_data.loc[sub_index,'hm_num_use_total'] = hm_df.shape[0]
    summary_data.loc[sub_index,'hm_lat_total'] = subject_df['lat'].mean()
    summary_data.loc[sub_index,'hm_lat_total_std'] = hm_df['lat'].std()
    summary_data.loc[sub_index,'hm_num_use_neut'] = sub_neut.shape[0]
    summary_data.loc[sub_index,'hm_mean_lat_neut'] = sub_neut['lat'].mean()
    summary_data.loc[sub_index,'hm_std_lat_neut'] = sub_neut['lat'].std()
    summary_data.loc[sub_index,'hm_num_use_pun'] = sub_pun.shape[0]
    summary_data.loc[sub_index,'hm_mean_lat_pun'] = sub_pun['lat'].mean()
    summary_data.loc[sub_index,'hm_std_lat_pun'] = sub_pun['lat'].std()
    summary_data.loc[sub_index,'hm_mot_pun'] = sub_pun['motion'].mean()
    summary_data.loc[sub_index,'hm_mot_neut'] = sub_neut['motion'].mean()
    summary_data.loc[sub_index,'hm_mot_all'] = hm_df['motion'].mean()
    summary_data.loc[sub_index,'hm_acc_all'] = hm_df['acc'].mean()
    summary_data.loc[sub_index,'hm_acc_pun'] = sub_pun['acc'].mean()
    summary_data.loc[sub_index,'hm_acc_neut'] = sub_neut['acc'].mean()
    
summary_data.to_csv(behavior_dir + 'fullsample_means.csv')
