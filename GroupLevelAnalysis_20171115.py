
# coding: utf-8

# In[ ]:

# Import stuff
from os.path import join
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.fsl.utils import Merge, ImageMeants
from nipype.interfaces.fsl.model import Randomise, Cluster
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.fsl.maths import ApplyMask
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
MNI_template = template_dir + '/MNI152_T1_1mm_brain.nii'

#pull subject info to iter over
subject_info = DataFrame.from_csv(analysis_home + '/misc/subjs.csv')
subjects_list = subject_info['SubjID'].tolist()
timepoints = subject_info['Timepoint'].tolist()

conditions = ['punish','neutral']
seed_names = ['L_amyg','R_amyg']

# Group analysis files
MCage = analysis_home + '/misc/MCage_groups.mat'
MCageSq = analysis_home + '/misc/MCageSq_groups.mat'
nonDevSex = analysis_home + '/misc/NonDevSex_groups.mat'
MCage_con = analysis_home + '/misc/tcon_MCage.con'
MCageSq_con = analysis_home + '/misc/tcon_MCageSq.con'
nonDevSex_con = analysis_home + '/misc/tcon_NonDevSex.con'
MCageInv = analysis_home + '/misc/MCageInv_groups.mat'
MCageInv_con = analysis_home + '/misc/tcon_MCageInv.con'

group_designs = [MCage, MCageSq, nonDevSex, MCageInv]
contrasts = [MCage_con, MCageSq_con, nonDevSex_con, MCageInv_con]


# In[ ]:

## Data handling nodes

conditionsource = Node(IdentityInterface(fields=['condition','seed']),
                       name='conditionsource')
conditionsource.iterables = [('condition',conditions),('seed', seed_names)]

# Grab the subject beta maps 
time_template = {'beta_maps':firstlevel_dir + '/smoothedMNI_conn_beta/*/%s/%s/betas_flirt_smooth_masked.nii'}
betamap_grabber = Node(DataGrabber(sort_filelist=True,
                                   field_template = time_template,
                                   base_directory=firstlevel_dir,
                                   template=firstlevel_dir + '/smoothedMNI_conn_beta/*/%s/%s/betas_flirt_smooth_masked.nii',
                                   infields=['condition','seed'],
                                   template_args={'beta_maps':[['condition','seed']]}), 
                       name='betamap_grabber')

# Sink relavent data
substitutions = [('_condition_',''),
                 ('_seed_',''), 
                 ('_design_mat_..Volumes..Zeus..Cat..misc..',''), 
                 ('_tcon_..Volumes..Zeus..Cat..misc..tcon_','')]
datasink = Node(DataSink(substitutions=substitutions, 
                         base_directory=secondlevel_dir,
                         container=secondlevel_dir), 
                name='datasink')


# In[ ]:

## Analysis nodes

# merge beta maps into one file
merge = Node(Merge(dimension = 't'),name='merge')

# Carry out t tests with permutation testing
randomise = Node(Randomise(tfce=True,  
                           num_perm=1000),
                 name='randomise')
randomise.iterables = [('design_mat',group_designs), ('tcon', contrasts)]
randomise.synchronize = True

# Threshold the t corrected p files
binarize_pmap = MapNode(Binarize(min=0.95), name = 'binarize_pmap', iterfield=['in_file'])

mask_tstat = MapNode(ApplyMask(),name='mask_tstat', iterfield=['in_file','mask_file'])

# Cluster the results
cluster_results = MapNode(Cluster(threshold=2,
                                  out_index_file=True,
                                  out_localmax_txt_file=True),
                          name='cluster_results', 
                          iterfield = ['in_file'])


# In[ ]:

groupanalysisflow = Workflow(name='groupanalysisflow')
groupanalysisflow.connect([(conditionsource, betamap_grabber, [('condition','condition'),
                                                               ('seed','seed')]),
                           (betamap_grabber, merge, [('beta_maps','in_files')]),
                           (merge, randomise, [('merged_file','in_file')]),
                           (randomise, binarize_pmap, [('t_corrected_p_files','in_file')]),
                           (binarize_pmap, mask_tstat, [('binary_file','mask_file')]),
                           (randomise, mask_tstat, [('tstat_files','in_file')]),
                           (mask_tstat, cluster_results, [('out_file','in_file')]),
                           
                           (randomise, datasink, [('t_corrected_p_files','t_corrected_p_files'),
                                                  ('tstat_files','tstat_files')]),
                           (cluster_results, datasink, [('index_file','cluster_index_file'), 
                                                        ('localmax_txt_file','cluster_localmax_txt_file')])
                          ])
groupanalysisflow.base_dir = workflow_dir
groupanalysisflow.write_graph(graph2use='flat')
groupanalysisflow.run('MultiProc', plugin_args={'n_procs':24})

