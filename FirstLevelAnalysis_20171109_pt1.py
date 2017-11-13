
# coding: utf-8

# In[ ]:

# Import stuff
from os.path import join
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.fsl.preprocess import FLIRT, SUSAN
from nipype.interfaces.fsl.utils import Merge, ImageMeants
from nipype.interfaces.fsl.model import GLM, Level1Design, FEATModel, FILMGLS
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.freesurfer.model import Binarize
from pandas import DataFrame, Series

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12/toolbox')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# FSL set up- change default file output type
from nipype.interfaces.fsl import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

# Set study variables
#analysis_home = '/Users/catcamacho/Box/LNCD_rewards_connectivity'
analysis_home = '/Volumes/Zeus/Cat'
#raw_dir = analysis_home + '/subjs'
raw_dir = '/Volumes/Phillips/bars/APWF_bars/subjs'
preproc_dir = analysis_home + '/proc/preprocessing'
firstlevel_dir = analysis_home + '/proc/firstlevel'
secondlevel_dir = analysis_home + '/proc/secondlevel'
workflow_dir = analysis_home + '/workflows'
template_dir = analysis_home + '/templates'

MNI_template = template_dir + '/MNI152_T1_2mm_brain.nii'
#pull subject info to iter over
subject_info = DataFrame.from_csv(analysis_home + '/misc/subjs.csv')
subjects_list = subject_info['SubjID'].tolist()
timepoints = subject_info['Timepoint'].tolist()

#subjects_list = [10766]
#timepoints = [1]

# Seeds list- based on aseg segmentation
L_amyg = 18
R_amyg = 54

seeds = [L_amyg, R_amyg]
seed_names = ['L_amyg','R_amyg']

TR = 1.5

conditions = ['punish','reward','neutral']
motion_thresh = 0.9 #in millimeters for trial-wise exclusion of data
BOLD_window = 8 # in TRs
smoothing_kernel = 6


# In[ ]:

# Data handling nodes
infosource = Node(IdentityInterface(fields=['subjid','timepoint']), 
                  name='infosource')
infosource.iterables = [('subjid', subjects_list),('timepoint', timepoints)]
infosource.synchronize = True

#grab timing files
time_template = {'timing':raw_dir + '/%s/%d_*/timing/*score_timing.txt'}
timegrabber = Node(DataGrabber(sort_filelist=True,
                               template = raw_dir + '/%d/%d_*/timing/*score_timing.txt',
                               field_template = time_template,
                               base_directory=raw_dir,
                               infields=['subjid','timepoint'], 
                               template_args={'timing':[['subjid','timepoint']]}), 
                   name='timegrabber')

# Grab niftis
template = {'struct': preproc_dir + '/preproc_anat/{subjid}_t{timepoint}/reoriented_anat.nii',
            'func': preproc_dir + '/preproc_func/{subjid}_t{timepoint}/func_filtered.nii',
            'segmentation': preproc_dir + '/aseg/{subjid}_t{timepoint}/reoriented_aseg.nii',
            'motion': preproc_dir + '/motion_params/{subjid}_t{timepoint}/allmotion.txt'}
datasource = Node(SelectFiles(template), 
                  name = 'datasource')

#sink important data
substitutions = [('_subjid_', ''),
                 ('_timepoint_','_t'), 
                 ('_condition_','')]
datasink = Node(DataSink(substitutions=substitutions, 
                         base_directory=firstlevel_dir,
                         container=firstlevel_dir), 
                name='datasink')


# In[2]:

# Extract timing for Beta Series Method- mark trials as high and low motion
def timing_bars(run_timing_list, condition, motion, motion_thresh, BOLD_window):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from pandas import DataFrame,Series,read_table,concat
    from nipype.interfaces.base import Bunch
    
    # Import and organize motion data
    motion_df = read_table(motion,delim_whitespace=True,header=None)
    mean_translation = motion_df[[3,4,5]].mean(axis=1)
    
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
    df_full = df_full.sort(['runNum','time_hyp'], ascending=[1,1])
    df_full.loc[:,'motion'] = mean_translation
    
    # Sort out trials that are both complete and received a response
    df_responded = df_full[df_full.loc[:,'Count'] == 1]
    df_responded = df_responded[df_responded.loc[:,'catch']==0]

    # Sort out trial onsets for the condition of interest
    df_condition = df_responded[df_responded.loc[:,'cond']==condition]
    df_condition = df_condition[df_condition.loc[:,'stim']=='cue']
    
    # Add additional label to the trials with high motion
    df_condition.loc[:,'mot_cat'] = Series('low',index=df_condition.index)
    for index, row in df_condition.iterrows():
        hrf_length = index+BOLD_window
        trial_motion = df_full.iloc[index:hrf_length,8]
        excess_vols = (trial_motion >= motion_thresh) + (trial_motion <= (-1*motion_thresh))
        if sum(excess_vols) >= 4:
            df_condition.loc[index,'mot_cat'] = 'high'    
    
    lowmotion = df_condition[df_condition.loc[:, 'mot_cat'] == 'low']
    highmotion = df_condition[df_condition.loc[:, 'mot_cat'] == 'high']
    
    # create onsets list
    lm_onsets = lowmotion['time_hyp'].tolist()
    hm_onsets = highmotion['time_hyp'].tolist()
    lm_onsets_list = [[o] for o in lm_onsets]
    hm_onsets_list = [[p] for p in hm_onsets]
    onsets = lm_onsets_list + hm_onsets_list
    
    # create trial names
    lm_trialnames = [(condition + '_lm' + str(h)) for h in range(0,len(lm_onsets_list))]
    hm_trialnames = [(condition + '_hm' + str(i)) for i in range(0,len(hm_onsets_list))]
    trialNames = lm_trialnames + hm_trialnames
    
    #make bunch file
    timing_bunch = []
    timing_bunch.insert(0,Bunch(conditions=trialNames,
                                onsets=onsets,
                                durations=[[4.5] for s in trialNames],
                                amplitudes=None,
                                tmod=None,
                                pmod=None,
                                regressor_names=None,
                                regressors=None))
    return(timing_bunch)

# Function to create contrast lists from a bunch file
def beta_contrasts(timing_bunch):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from nipype.interfaces.base import Bunch
    from numpy import zeros
    
    conditions_names = timing_bunch[0].conditions
    
    # Make the contrast vectors for each trial
    boolean_con_lists = []
    num_cons = len(conditions_names)
    for i in range(0,num_cons):
        boo = zeros(num_cons)
        boo[i] = 1
        boolean_con_lists.append(list(boo))
    
    # Create the list of lists for the full contrast info
    contrasts_list = []
    for a in range(0,num_cons):
        con = [conditions_names[a], 'T', conditions_names, boolean_con_lists[a]]
        contrasts_list.append(con)
        
    return(contrasts_list)

# Function to write the beta series trial names to a text file
def beta_list(timing_bunch):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from nipype.interfaces.base import Bunch
    from os.path import abspath
    
    conditions_names = timing_bunch[0].conditions
    filename = open('beta_condition_names.txt','w')
    for line in conditions_names:
        filename.write(line + '\n')
    
    filename.close()
    condition_file = abspath('beta_condition_names.txt')
    
    return(condition_file)


# In[ ]:

# Extract timing
pull_timing = Node(Function(input_names=['run_timing_list','condition',
                                         'motion','motion_thresh','BOLD_window'],
                            output_names=['timing_bunch'],
                            function=timing_bars), name='pull_timing')
pull_timing.inputs.BOLD_window = BOLD_window
pull_timing.inputs.motion_thresh = motion_thresh
pull_timing.iterables = [('condition',conditions)]

# create the list of T-contrasts
define_contrasts = Node(Function(input_names=['timing_bunch'], 
                                 output_names = ['contrasts_list'], 
                                 function=beta_contrasts),
                        name = 'define_contrasts')

# Save the beta series list name for later organization
save_beta_list = Node(Function(input_names=['timing_bunch'], 
                               output_names=['condition_file'], 
                               function=beta_list), 
                      name='save_beta_list')

# Specify FSL model - input bunch file called subject_info
modelspec = Node(SpecifyModel(time_repetition=TR, 
                              input_units='secs',
                              high_pass_filter_cutoff=0),
                 name='modelspec')

# Generate a level 1 design
level1design = Node(Level1Design(bases={'dgamma':{'derivs': False}},
                                 interscan_interval=TR, # the TR
                                 model_serial_correlations=True), 
                    name='level1design')

# Estimate Level 1
generateModel = Node(FEATModel(), 
                     name='generateModel')

# Run GLM
extract_betas = Node(FILMGLS(threshold=-1000, 
                             fit_armodel=False,
                             smooth_autocorr=False,
                             full_data=True), 
                     name='extract_betas')


# In[ ]:

# Connect the workflow
betaseriesflow = Workflow(name='betaseriesflow')
betaseriesflow.connect([(infosource, datasource,[('subjid','subjid')]),
                        (infosource, datasource,[('timepoint','timepoint')]),
                        (infosource, timegrabber,[('subjid','subjid')]),
                        (infosource, timegrabber,[('timepoint','timepoint')]),
                        (timegrabber, pull_timing, [('timing','run_timing_list')]),
                        (datasource, pull_timing, [('motion','motion')]),
                        (pull_timing, modelspec, [('timing_bunch','subject_info')]),
                        (datasource, modelspec, [('func','functional_runs')]),
                        (pull_timing, define_contrasts, [('timing_bunch','timing_bunch')]),
                        (define_contrasts, level1design, [('contrasts_list','contrasts')]),
                        (modelspec, level1design, [('session_info','session_info')]),
                        (level1design,generateModel, [('ev_files','ev_files')]),
                        (level1design,generateModel, [('fsf_files','fsf_file')]),
                        (generateModel,extract_betas, [('design_file','design_file')]),
                        (generateModel,extract_betas, [('con_file','tcon_file')]),
                        (datasource,extract_betas, [('func','in_file')]),
                        (pull_timing,save_beta_list, [('timing_bunch','timing_bunch')]),
                        
                        (save_beta_list,datasink, [('condition_file','condition_file')]),
                        (extract_betas,datasink,[('copes','copes')]),
                        (extract_betas,datasink,[('param_estimates','betas')]),
                        (extract_betas,datasink,[('tstats','tstats')]),
                        (generateModel,datasink,[('design_image','design_image')])
                       ])
betaseriesflow.base_dir = workflow_dir
betaseriesflow.write_graph(graph2use='flat')
betaseriesflow.run('MultiProc', plugin_args={'n_procs': 30})



