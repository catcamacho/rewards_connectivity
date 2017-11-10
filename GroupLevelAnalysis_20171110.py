
# coding: utf-8

# In[ ]:

# Import stuff
from os.path import join
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.fsl.utils import Merge, ImageMeants
from nipype.interfaces.fsl.model import Randomise
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

conditions = ['punish','reward','neutral']
seed_names = ['L_amyg','R_amyg']

# Group analysis files
group_design = analysis_home + '/misc/groups.mat'
t_contrasts = analysis_home + '/misc/tcon.con'


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
                 ('_seed_','')]
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
                           design_mat=group_design, 
                           tcon=t_contrasts, 
                           num_perm=500),
                 name='randomise')

# Threshold the t corrected p files- add when I need it



# In[ ]:

groupanalysisflow = Workflow(name='groupanalysisflow')
groupanalysisflow.connect([(conditionsource, betamap_grabber, [('condition','condition')]),
                           (conditionsource, betamap_grabber, [('seed','seed')]),
                           (betamap_grabber, merge, [('beta_maps','in_files')]),
                           (merge, randomise, [('merged_file','in_file')]),
                           
                           (randomise, datasink, [('t_corrected_p_files','t_corrected_p_files')]),
                           (randomise, datasink, [('tstat_files','tstat_files')])
                          ])
groupanalysisflow.basedir = workflow_dir
groupanalysisflow.write_graph(graph2use='flat')
groupanalysisflow.run('Multiproc', plugin_args={'n_procs':2})

