
# coding: utf-8

# In[ ]:

# Import stuff
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.fsl.preprocess import FLIRT, SUSAN
from nipype.interfaces.fsl.utils import Merge, ImageMeants
from nipype.interfaces.fsl.model import GLM, Level1Design, FEATModel, FILMGLS
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.nipy.model import FitGLM, EstimateContrast
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

MNI_template = template_dir + '/MNI152_T1_3mm_brain.nii'
MNI_mask = template_dir + '/MNI152_T1_3mm_mask.nii'

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
min_trials_for_usability = 20 #per condition, selected based on Paulsen 2015


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

# FD motion grabber
motion_template = {'motion':preproc_dir + '/FD_out_metric_values/%s_t%d/*/FD.txt'}
motiongrabber = Node(DataGrabber(sort_filelist=True,
                                 template = preproc_dir + '/FD_out_metric_values/%s_t%d/*/FD.txt',
                                 field_template = motion_template,
                                 base_directory=preproc_dir,
                                 infields=['subjid','timepoint'], 
                                 template_args={'motion':[['subjid','timepoint']]}),
                     name='motiongrabber')

# Grab niftis
template = {'struct': preproc_dir + '/preproc_anat/{subjid}_t{timepoint}/reoriented_anat.nii',
            'func': preproc_dir + '/preproc_func/{subjid}_t{timepoint}/func_filtered.nii',
            'segmentation': preproc_dir + '/aseg/{subjid}_t{timepoint}/reoriented_aseg.nii'}
datasource = Node(SelectFiles(template), 
                  name = 'datasource')

#sink important data
substitutions = [('_subjid_', ''),
                 ('_timepoint_','_t'), 
                 ('_condition_',''),
                 ('_max_18_min_18','L_amyg'), 
                 ('_max_54_min_54','R_amyg')]
datasink = Node(DataSink(substitutions=substitutions, 
                         base_directory=firstlevel_dir,
                         container=firstlevel_dir), 
                name='datasink')


# In[ ]:

# Extract timing for Beta Series Method- mark trials as high and low motion
def timing_bars(run_timing_list, motion, motion_thresh, BOLD_window):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from pandas import DataFrame,Series,read_table,concat
    from nipype.interfaces.base import Bunch
    
    # Import and organize motion data
    motion_dfs = [ read_table(j,delim_whitespace=True,header=None) for j in motion ]
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
    
    # Sort out trials that are both complete and received a response
    #df_responded = df_full[df_full.loc[:,'Count'] == 1]
    #df_responded = df_responded[df_responded.loc[:,'catch']==0]

    # Sort out trial onsets for the condition of interest
    #df_condition = df_responded[df_responded.loc[:,'cond']==condition]
    df_trial = df_full[df_full.loc[:,'stim']=='cue']
    
    # Add additional label to the trials with high motion
    df_trial.loc[:,'mot_cat'] = Series('low',index=df_trial.index)
    for index, row in df_trial.iterrows():
        hrf_length = index+BOLD_window
        trial_motion = df_full.iloc[index:hrf_length,8]
        excess_vols = (trial_motion >= motion_thresh)
        if sum(excess_vols) >= 3:
            df_trial.loc[index,'mot_cat'] = 'high'    
    
    df_fulltrials = df_trial[df_trial.loc[:, 'catch'] == 0]
    df_responded = df_fulltrials[df_fulltrials.loc[:, 'Count'] == 1]
    
    lowmotion = df_responded[df_trial.loc[:, 'mot_cat'] == 'low']
    highmotion = df_responded[df_trial.loc[:, 'mot_cat'] == 'high']
    catch1 = df_trial[df_trial.loc[:, 'catch'] == 1]
    catch2 = df_trial[df_trial.loc[:, 'catch'] == 2]
    
    # create onsets list
    lm_onsets = lowmotion['time_hyp'].tolist()
    hm_onsets = highmotion['time_hyp'].tolist()
    catch1_onsets = catch1['time_hyp'].tolist()
    catch2_onsets = catch2['time_hyp'].tolist()
    lm_onsets_list = [[o] for o in lm_onsets]
    hm_onsets_list = [[p] for p in hm_onsets]
    c1_onsets_list = [[q] for q in catch1_onsets]
    c2_onsets_list = [[r] for r in catch2_onsets]
    onsets = lm_onsets_list + hm_onsets_list + c1_onsets_list + c2_onsets_list
    
    # create trial names
    lm_conditions = lowmotion['cond'].tolist()
    hm_conditions = highmotion['cond'].tolist()
    lm_trialnames = [(lm_conditions[h] + '_lm' + str(h)) for h in range(0,len(lm_onsets_list))]
    hm_trialnames = [(hm_conditions[i] + '_hm' + str(i)) for i in range(0,len(hm_onsets_list))]
    catch1_trialnames = [('catch1_' + str(j)) for j in range(0,len(c1_onsets_list))]
    catch2_trialnames = [('catch2_' + str(k)) for k in range(0,len(c2_onsets_list))]
    
    trialNames = lm_trialnames + hm_trialnames + catch1_trialnames + catch2_trialnames
    
    # creat durations list
    fulltrials = lm_trialnames + hm_trialnames
    fulltrial_durations = [[4.5] for s in fulltrials]
    c1_durations = [[1.5] for t in catch1_trialnames]
    c2_durations = [[3.0] for u in catch2_trialnames]
    durations = fulltrial_durations + c1_durations + c2_durations
    
    #make bunch file
    timing_bunch = []
    timing_bunch.insert(0,Bunch(conditions=trialNames,
                                onsets=onsets,
                                durations=durations,
                                amplitudes=None,
                                tmod=None,
                                pmod=None,
                                regressor_names=None,
                                regressors=None))
    return(timing_bunch)

def hrf_fitting_onsets(timing_bunch):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from nipype.interfaces.base import Bunch
    from pandas import DataFrame,Series,read_table
    
    conditions_names = timing_bunch[0].conditions
    onsets = [a for [a] in timing_bunch[0].onsets]
    durations = [b for [b] in timing_bunch[0].durations]
    
    #remove catch trials
    num_cons = len(conditions_names)
    r = 0
    for i in range(0,num_cons):
        if 'catch' in conditions_names[i]:
            del onsets[i-r]
            del durations[i-r]
            r = r + 1
        elif '_hm' in conditions_names[i]:
            del onsets[i-r]
            del durations[i-r]
            r = r + 1
    
    hrf_fit_bunch = []
    hrf_fit_bunch.insert(0,Bunch(conditions=['HR_Event'],
                                 onsets=[onsets],
                                 durations=[durations],
                                 amplitudes=None,
                                 tmod=None,
                                 pmod=None,
                                 regressor_names=None,
                                 regressors=None))
    return(hrf_fit_bunch)

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
    filename = open('betanames.txt','w')
    for line in conditions_names:
        filename.write(line + '\n')
    
    filename.close()
    condition_file = abspath('betanames.txt')
    
    return(condition_file)


# In[ ]:

# Extract timing
pull_timing = Node(Function(input_names=['run_timing_list','motion',
                                         'motion_thresh','BOLD_window'],
                            output_names=['timing_bunch'],
                            function=timing_bars), name='pull_timing')
pull_timing.inputs.BOLD_window = BOLD_window
pull_timing.inputs.motion_thresh = motion_thresh

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
#level1design = Node(Level1Design(bases={'dgamma':{'derivs': False}},
#                                 interscan_interval=TR, # the TR
#                                 model_serial_correlations=True), 
#                    name='level1design')

# Estimate Level 1
#generateModel = Node(FEATModel(), 
#                     name='generateModel')

# Run GLM
#extract_betas = Node(FILMGLS(threshold=-1000, 
#                             fit_armodel=False,
#                             smooth_autocorr=False,
#                             full_data=True), 
#                     name='extract_betas')

extract_betas = Node(FitGLM(TR=TR,
                            plot_design_matrix=True,
                            save_residuals=True,
                            drift_model ='Polynomial'), 
                     name='extract_betas')


# In[ ]:

## Find voxels associated with task at any level. Use this as an HRF fit mask
# pull timing

hrffit_timing = Node(Function(input_names=['timing_bunch'], 
                              output_names=['hrf_fit_bunch'], 
                              function=hrf_fitting_onsets), 
                     name='hrffit_timing')

# create the list of T-contrasts
define_hrffit_con = Node(Function(input_names=['timing_bunch'], 
                                  output_names = ['contrasts_list'], 
                                  function=beta_contrasts),
                         name = 'define_hrffit_con')

# Specify FSL model - input bunch file called subject_info
hrffit_model = Node(SpecifyModel(time_repetition=TR,
                                 input_units='secs',
                                 high_pass_filter_cutoff=0),
                    name='hrffit_model')

# Generate a level 1 design
#hrffitdesign = Node(Level1Design(bases={'dgamma':{'derivs': False}},
#                                 interscan_interval=TR, # the TR
#                                 model_serial_correlations=True), 
#                    name='hrffitdesign')

# Estimate Level 1
#genHRFfitModel = Node(FEATModel(), 
#                      name='genHRFfitModel')

# Run GLM
#HRFfitting = Node(FILMGLS(threshold=-1000, 
#                          fit_armodel=False,
#                          smooth_autocorr=False,
#                          full_data=True), 
#                  name='HRFfitting')

#run GLM
HRFfitting = Node(FitGLM(TR=TR,
                         plot_design_matrix=True,
                         save_residuals=True, 
                         drift_model ='Polynomial'), 
                  name='HRFfitting')

est_contrast = Node(EstimateContrast(), 
                    name='est_contrast')

thresh_HRFfit_map = MapNode(Binarize(abs=True, 
                                     min=2),
                            name='thresh_HRFfit_map', 
                            iterfield = ['in_file'])


# In[ ]:

# Connect the workflow
betaseriesflow = Workflow(name='betaseriesflow')
betaseriesflow.connect([(infosource, datasource,[('subjid','subjid')]),
                        (infosource, datasource,[('timepoint','timepoint')]),
                        (infosource, timegrabber,[('subjid','subjid')]),
                        (infosource, timegrabber,[('timepoint','timepoint')]),
                        (infosource, motiongrabber,[('subjid','subjid')]),
                        (infosource, motiongrabber,[('timepoint','timepoint')]),
                        (timegrabber, pull_timing, [('timing','run_timing_list')]),
                        (motiongrabber, pull_timing, [('motion','motion')]),
                        (pull_timing, modelspec, [('timing_bunch','subject_info')]),
                        (datasource, modelspec, [('func','functional_runs')]),
                        #(pull_timing, define_contrasts, [('timing_bunch','timing_bunch')]),
                        #(define_contrasts, level1design, [('contrasts_list','contrasts')]),
                        #(modelspec, level1design, [('session_info','session_info')]),
                        #(level1design,generateModel, [('ev_files','ev_files')]),
                        #(level1design,generateModel, [('fsf_files','fsf_file')]),
                        #(generateModel,extract_betas, [('design_file','design_file')]),
                        #(generateModel,extract_betas, [('con_file','tcon_file')]),
                        (modelspec, extract_betas, [('session_info','session_info')]),
                        #(datasource,extract_betas, [('func','in_file')]),
                        (pull_timing,save_beta_list, [('timing_bunch','timing_bunch')]),
                        
                        (pull_timing, hrffit_timing,[('timing_bunch','timing_bunch')]),
                        (hrffit_timing, hrffit_model, [('hrf_fit_bunch','subject_info')]),
                        (datasource, hrffit_model, [('func','functional_runs')]),
                        (hrffit_timing, define_hrffit_con, [('hrf_fit_bunch','timing_bunch')]),
                        (define_hrffit_con, est_contrast, [('contrasts_list','contrasts')]),
                        (hrffit_model, HRFfitting, [('session_info','session_info')]),
                        #(hrffitdesign, genHRFfitModel, [('ev_files','ev_files')]),
                        #(hrffitdesign, genHRFfitModel, [('fsf_files','fsf_file')]),
                        #(genHRFfitModel, HRFfitting, [('design_file','design_file')]),
                        #(genHRFfitModel, HRFfitting, [('con_file','tcon_file')]),
                        #(datasource,HRFfitting, [('func','in_file')]),
                        (HRFfitting, est_contrast, [('axis','axis')]),
                        (HRFfitting, est_contrast, [('beta','beta')]),
                        (HRFfitting, est_contrast, [('constants','constants')]),
                        (HRFfitting, est_contrast, [('dof','dof')]),
                        (HRFfitting, est_contrast, [('nvbeta','nvbeta')]),
                        (HRFfitting, est_contrast, [('reg_names','reg_names')]),
                        (HRFfitting, est_contrast, [('s2','s2')]),
                        (est_contrast, thresh_HRFfit_map, [('stat_maps','in_file')]),
                        
                        (save_beta_list,datasink, [('condition_file','condition_file')]),
                        #(extract_betas,datasink,[('copes','copes')]),
                        (extract_betas,datasink,[('beta','betas')]),
                        #(extract_betas,datasink,[('tstats','tstats')]),
                        #(generateModel,datasink,[('design_image','design_image')]),
                        (HRFfitting,datasink,[('beta','HRFfit_betas')]),
                        (HRFfitting,datasink,[('a','HRFfit_a')]),
                        (est_contrast,datasink,[('stat_maps','HRFfit_tstats')]),
                        (est_contrast,datasink,[('p_maps','HRFfit_pmap')]),
                        (est_contrast,datasink,[('z_maps','HRFfit_zmaps')]),
                        (thresh_HRFfit_map, datasink, [('binary_file','HRFfitMask')])
                       ])
betaseriesflow.base_dir = workflow_dir
betaseriesflow.write_graph(graph2use='flat')
betaseriesflow.run('MultiProc', plugin_args={'n_procs': 20})

