
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
firstlevel_dir = analysis_home + '/proc/firstlevel'
secondlevel_dir = analysis_home + '/proc/secondlevel'
workflow_dir = analysis_home + '/workflows'
template_dir = analysis_home + '/templates'
MNI_template = template_dir + '/MNI152_T1_1mm_brain.nii'
MNI_mask = template_dir + '/MNI152_T1_3mm_mask.nii'


#pull subject info 
subject_info = analysis_home + '/misc/subjs.csv'

conditions = ['punish','neutral']
seed_names = ['L_amyg','R_amyg']

# Group analysis models (predicting FC)
models = ['brain ~ ageMC + sex + ageMC*sex', 
          'brain ~ invAgeMC + sex + invAgeMC*sex']


# In[ ]:

# LMEM for MRI data (3D nifti data)
def mri_lmem(model, mask, subject_dataframe, subject_files, grouping_variable):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)

    from os import getcwd
    from os.path import abspath
    import statsmodels.formula.api as smf
    from nibabel import load, save, Nifti1Image
    from numpy import array, empty_like, stack, nditer, zeros_like, zeros
    from pandas import DataFrame, read_csv, Series, concat
    from warnings import filterwarnings
    filterwarnings("ignore")

    working_dir = getcwd() + '/'
    subj_data = read_csv(subject_dataframe, header=0, index_col=0)

    # Load the brain data
    brain_niftis = load(subject_files)
    brain_data_4D = brain_niftis.get_data()

    # Load the mask
    mask_nifti = load(mask)
    mask = mask_nifti.get_data()

    ## Preallocate the output arrays
    # for the model
    BIC_data = zeros_like(mask).astype(float)
    AIC_data = zeros_like(mask).astype(float)
    pval_intercept_data = zeros_like(mask).astype(float)
    pval_age_data = zeros_like(mask).astype(float)
    pval_sex_data = zeros_like(mask).astype(float)
    pval_ageSexInteract_data = zeros_like(mask).astype(float)
    # per subject
    residuals_data = zeros_like(brain_data_4D).astype(float)
    pred_values_data = zeros_like(brain_data_4D).astype(float)

    # Set up the actual loops to pull in subject data and do the modeling
    for x in range(0,mask.shape[0]):
        for y in range(0,mask.shape[1]):
            for z in range(0,mask.shape[2]):
                if mask[x][y][z] == 1:
                    voxel = zeros(brain_data_4D.shape[3])
                    for a in range(0,brain_data_4D.shape[3]):
                        voxel[a] = brain_data_4D[x][y][z][a]
                    voxel = Series(voxel, index=subj_data.index, name='brain')
                    data = concat([voxel, subj_data],axis=1)
                    mlm = smf.mixedlm(model, data, groups=data[grouping_variable])
                    mod = mlm.fit()
                    pval_intercept_data[x][y][z] = 1 - mod.pvalues[0]
                    pval_age_data[x][y][z] = 1 - mod.pvalues[1]
                    pval_sex_data[x][y][z] = 1 - mod.pvalues[2]
                    pval_ageSexInteract_data[x][y][z] = 1 - mod.pvalues[3]
                    BIC_data[x][y][z] = mod.bic
                    AIC_data[x][y][z] = mod.aic
                    residuals = mod.resid
                    pred_values = Series(mod.predict(), index = subj_data.index)
                    for d in range(0,brain_data_4D.shape[3]):
                        residuals_data[x][y][z][d] = residuals.tolist()[d]
                        pred_values_data[x][y][z][d] = pred_values.tolist()[d]

                
    # Save the ouputs as nifti files
    output_data = [BIC_data, AIC_data, pval_intercept_data, pval_age_data,
                    pval_sex_data, pval_ageSexInteract_data, residuals_data, 
                    pred_values_data]
    output_niftis = [Nifti1Image(result, mask_nifti.affine) for result in output_data]
    
    output_filenames = ['BICs.nii','AICs.nii','pval_intercept_data.nii',
                        'pval_age_data.nii','pval_sex_data.nii',
                        'pval_ageSexInteract_data.nii','residuals_data.nii',
                        'pred_values_data.nii']
    for e in range(0,len(output_niftis)):
        save(output_niftis[e], working_dir + output_filenames[e])
    
    output_volumes = [abspath(output_filenames[0]),
                      abspath(output_filenames[1]),
                      abspath(output_filenames[2]), 
                      abspath(output_filenames[3]), 
                      abspath(output_filenames[4]), 
                      abspath(output_filenames[5]), 
                      abspath(output_filenames[6]), 
                      abspath(output_filenames[7])]
    
    return(output_volumes)



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
                 ('brain~ageMC+sex+ageMC*sex','linearAge'),
                 ('brain~invAgeMC+sex+invAgeMC*sex','inverseAge')]
datasink = Node(DataSink(substitutions=substitutions, 
                         base_directory=secondlevel_dir,
                         container=secondlevel_dir), 
                name='datasink')


# In[ ]:

## Analysis nodes

#Merge subject files together
merge = Node(Merge(dimension='t'), name='merge')

# Linear mixed effects modeling
lmemodel = Node(Function(input_names = ['model', 'mask', 'subject_dataframe', 
                                        'subject_files', 'grouping_variable'], 
                         output_names = ['output_volumes'], 
                         function=mri_lmem), 
                name='lmemodel')
lmemodel.iterables = [('model', models)]
lmemodel.inputs.mask = MNI_mask
lmemodel.inputs.subject_dataframe = subject_info
lmemodel.inputs.grouping_variable = 'Timepoint'

# Mask the file to only significant voxels for clustering
mask_stat = Node(Binarize(), name = 'mask_stat')

# Cluster the results
cluster_results = MapNode(Cluster(threshold=0.95,
                                  out_index_file=True,
                                  out_localmax_txt_file=True),
                          name='cluster_results', 
                          iterfield = ['in_file'])


# In[ ]:

LMEManalysisflow = Workflow(name='LMEManalysisflow')
LMEManalysisflow.connect([(conditionsource, betamap_grabber, [('condition','condition'),
                                                              ('seed','seed')]),
                          (betamap_grabber, merge, [('beta_maps','in_files')]),
                          (merge, lmemodel, [('merged_file','subject_files')]),
                          (lmemodel, datasink, [('output_volumes','output_volumes')])
                         ])
LMEManalysisflow.base_dir = workflow_dir
LMEManalysisflow.write_graph(graph2use='flat')
LMEManalysisflow.run('MultiProc', plugin_args={'n_procs':2})


# In[ ]:

#lme_template = 
#lme_datagrabber = 


# In[ ]:

#clusterflow = Workflow(name='clusterflow')
#clusterflow.connect([(mask_stat, cluster_results, [('out_file','in_file')]),
#                     (cluster_results, datasink, [('index_file','cluster_index_file'), 
#                                                  ('localmax_txt_file','cluster_localmax_txt_file')])])

