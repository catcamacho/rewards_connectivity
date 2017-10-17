
# coding: utf-8

# In[ ]:

# Import stuff
from os.path import join
from pandas import DataFrame

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataSink, DataGrabber, FreeSurferSource
from nipype.algorithms.misc import Gunzip

from nipype.interfaces.freesurfer.preprocess import ReconAll, MRIConvert
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.freesurfer import FSCommand
from nipype.interfaces.fsl.utils import Reorient2Std, Merge
from nipype.interfaces.fsl.preprocess import MCFLIRT, SliceTimer, FLIRT, FAST
from nipype.interfaces.fsl.maths import ApplyMask
#from nipype.interfaces.fsl.ICA_AROMA import ICA_AROMA
from nipype.algorithms.rapidart import ArtifactDetect
from nipype.interfaces.fsl.model import GLM
from nipype.algorithms.confounds import CompCor

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

subject_info = DataFrame.from_csv(analysis_home + '/misc/subjs.csv')
subjects_list = subject_info['SubjID'].tolist()
timepoints = subject_info['Timepoint'].tolist()

# FreeSurfer set up - change SUBJECTS_DIR 
fs_dir = analysis_home + '/proc/freesurfer'
FSCommand.set_default_subjects_dir(fs_dir)

# data collection specs
TR = 1.5 #in seconds
num_slices = 29
slice_direction = 3 #3 = z direction
interleaved = False
#all rates are in Hz (1/TR or samples/second)
highpass_freq = 0.01 #in Hz
lowpass_freq = 1 #in Hz


# In[ ]:

# Data handling nodes
infosource = Node(IdentityInterface(fields=['subjid','timepoint']), 
                  name='infosource')
infosource.iterables = [('subjid', subjects_list),('timepoint', timepoints)]
infosource.synchronize = True

#grab niftis
func_template = {'func':raw_dir + '/%s/%d_*/*/functional/functional.nii.gz'}
funcgrabber = Node(DataGrabber(sort_filelist=True,
                               template = raw_dir + '/%s/%d_*/*/functional/functional.nii.gz',
                               field_template = func_template,
                               base_directory=raw_dir,
                               infields=['subjid','timepoint'], 
                               template_args={'func':[['subjid','timepoint']]}), 
                   name='funcgrabber')

struct_template = {'struct':raw_dir + '/%s/%d_*/mprage/mprage.nii.gz'}
structgrabber = Node(DataGrabber(sort_filelist=True,
                                 template = raw_dir + '/%s/%d_*/mprage/mprage.nii.gz',
                                 field_template = struct_template,
                                 base_directory=raw_dir,
                                 infields=['subjid','timepoint'], 
                                 template_args={'struct':[['subjid','timepoint']]}), 
                     name='structgrabber')

substitutions = [('_subjid_', ''),
                 ('_timepoint_','_t')]
datasink = Node(DataSink(substitutions=substitutions, 
                         base_directory=preproc_dir,
                         container=preproc_dir), 
                name='datasink')


# In[ ]:

# Custom functions for anatomical processing
def numberedsub_convert(subjid,timepoint):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    fs_subjid = 's' + str(subjid) + '_' + str(timepoint)
    return(fs_subjid)


# In[ ]:

# Structural processing
get_fsID = Node(Function(input_names=['subjid','timepoint'],
                         output_names=['fs_subjid'],
                         function=numberedsub_convert),
                name='get_fsID')

# Use autorecon1 to skullstrip inputs: T1_files and subject_id; output: brainmask
fs_preproc = Node(ReconAll(directive='autorecon1',
                           flags='-gcut', 
                           openmp=4), 
                  name='fs_preproc')

# simultaneously convert to nifti and reslice inputs: in_file outputs: out_file
convert_anat = Node(MRIConvert(vox_size=(3,3,3), 
                               in_type='mgz',
                               out_file='anat.nii',
                               out_type='nii'), 
                    name='convert_anat')

# reorient to standard space inputs: in_file, outputs: out_file
reorient_anat = Node(Reorient2Std(out_file='reoriented_anat.nii',
                                  output_type='NIFTI'), 
                     name='reorient_anat')

# binarize anat, dilate 2 and erode 1 to fill gaps. Inputs: in_file; outputs: binary_file
binarize_anat = Node(Binarize(dilate=2,
                              erode=1, 
                              min=1,
                              max=300), 
                     name='binarize_anat')


# In[ ]:

# Custom functions referenced in the pipeline

# this function demeans each run and sorts the file list
def intensitycorrect(func_files):
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nipype.interfaces.fsl.utils import ImageMaths
    
    new_files = []
    n = 1
    for func in func_files:
        out_file = 'demeaned_func' + str(n)+ '.nii'
        math=ImageMaths()
        math.inputs.in_file = func
        math.inputs.out_file = out_file
        math.inputs.op_string = '-Tmean -mul -1 -add %s' % func
        math.run()
        demeaned_vol = abspath(out_file)
        new_files.append(demeaned_vol)
    
    new_func_list = sorted(new_files)
    return(new_func_list)

def concatenatemotion(motion_files,merged_func):
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from numpy import genfromtxt, vstack, savetxt
    from nibabel import load
    from warnings import warn
    
    func = load(merged_func)
    func_length = func.shape[3]
    
    n = 1
    for file in motion_files:
        temp = genfromtxt(file)
        if n == 1:
            all_motion=temp
        else:
            all_motion=vstack((all_motion,temp))
        n=n+1
    
    if func_length == all_motion.shape[0]:
        newmotion_file = 'allmotion.txt'
        savetxt(newmotion_file,all_motion)
        newmotion_params = abspath(newmotion_file)
    else:
        warn('The dimensions from your motion outputs do not match your functional data!')
        newmotion_file = 'allmotion.txt'
        savetxt(newmotion_file,all_motion)
        newmotion_params = abspath(newmotion_file)

    return(newmotion_params)

def bandpass_filter(in_file, lowpass, highpass, TR):
    import numpy as np
    import nibabel as nb
    from os.path import abspath
    from os import getcwd
    from nipype.interfaces.afni.preprocess import Bandpass
    from nipype.interfaces.afni.utils import AFNItoNIFTI
    from nipype.algorithms.misc import Gunzip
    from glob import glob
    from subprocess import call
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    out_file = 'func_filtered'
    call(['3dBandpass', '-prefix', out_file,'-dt', TR, highpass, lowpass, in_file])
    filtered_file = glob(getcwd() + '/func_filtered*.BRIK*')
    call(["gunzip", filtered_file[0]])
    new_file = getcwd() +'/func_filtered+orig.BRIK'
    
    cvt = AFNItoNIFTI()
    cvt.inputs.in_file = new_file
    cvt.inputs.outputtype = 'NIFTI'
    cvt.run()
    
    nii_file = glob(getcwd() + '/*.nii')
    out_file = nii_file[0]
    return(out_file)


# In[ ]:

# Functional processing

unzip_func = MapNode(Gunzip(), 
                     name='unzip_func', 
                     iterfield=['in_file'])

# Reorient each functional run in_file, out_file
reorient_func = MapNode(Reorient2Std(out_file='reoriented_func.nii'),
                        name='reorient_func', 
                        iterfield=['in_file'])

# Realign each volume to first volume in each run: in_file; out_file, par_file
realign_runs = MapNode(MCFLIRT(out_file='rfunc.nii',
                               save_plots=True,
                               save_rms=True), 
                       name='realign_runs',
                       iterfield=['in_file'])

# Slice time correction: in_file, slice_time_corrected_file
slicetime = MapNode(SliceTimer(time_repetition=TR, 
                               interleaved=interleaved, 
                               slice_direction=slice_direction, 
                               out_file='stfunc.nii'), 
                    name='slicetime',
                    iterfield=['in_file'])

# register the functional volumes to the subject space anat
# inputs: in_file, reference; out_file out_matrix_file
reg_func_to_anat = MapNode(FLIRT(out_matrix_file='xform.mat'),
                           name='reg_func_to_anat', 
                           iterfield=['in_file'])

apply_reg_to_func = MapNode(FLIRT(apply_xfm=True, 
                               out_file='warped_func.nii'), 
                            name='apply_reg_to_func', 
                            iterfield=['in_file','in_matrix_file'])

# Despiking and Intensity norm?
norm_run_intensities = Node(Function(input_names=['func_files'], 
                                     output_names=['new_func_list'],
                                     function=intensitycorrect),
                            name='norm_run_intensities')

# Merge the motion params into one long motion file: 
            # motion_files, merged_func; newmotion_params
merge_motion = Node(Function(input_names=['motion_files','merged_func'], 
                             output_names=['newmotion_params'], 
                             function=concatenatemotion), 
                    name='merge_motion')

# Merge all 4 runs: in_files, merged_file
merge_func = Node(Merge(dimension='t',
                        merged_file='merged_func.nii'),
                  name='merge_func')

# Realign each volume to first volume: in_file; out_file, par_file
realign_merged = Node(MCFLIRT(out_file='rmerged.nii',
                              ref_vol=0), 
                      name='realign_merged')

# Apply binary mask to merged functional scan: in_file, mask_file; out_file
mask_func = Node(ApplyMask(out_file='masked_func.nii'), 
                 name='mask_func')

# Bandpass Filtering (0.01-0.1 per Rissman et al 2004) all rates are in Hz (1/TR or samples/second)
bandpass = Node(name='bandpass', 
                interface=Function(input_names=['in_file','lowpass','highpass','TR'], 
                                   output_names=['out_file'],
                                   function=bandpass_filter))
bandpass.inputs.lowpass = lowpass_freq
bandpass.inputs.highpass = highpass_freq
bandpass.inputs.TR = TR


# In[ ]:

# Denoising
def adjust_masks(masks):
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from nipype.interfaces.freesurfer.model import Binarize
    #pve0 = csf, pve1 = gm, pve2 = wm
    
    origvols = sorted(masks)
    csf = origvols[0]
    wm = origvols[2]
    vols = []
    
    binary = Binarize()
    binary.inputs.in_file = wm
    binary.inputs.min = 0.5
    binary.inputs.max = 2
    binary.inputs.binary_file = 'WM_seg.nii'
    binary.run()
    wm_new = abspath(binary.inputs.binary_file)
    vols.append(wm_new)
    
    binary2 = Binarize()
    binary2.inputs.in_file = csf
    binary2.erode = 1
    binary2.inputs.min = 0.5
    binary2.inputs.max = 2
    binary2.inputs.binary_file = 'CSF_seg.nii'
    binary2.run()
    csf_new = abspath(binary2.inputs.binary_file)
    vols.append(csf_new)
    
    return(vols)
    
def create_noise_matrix(vols_to_censor,motion_params,comp_noise):
    from numpy import genfromtxt, zeros, column_stack, savetxt
    from os import path
    
    motion = genfromtxt(motion_params, delimiter=None, dtype=None, skip_header=0)
    comp_noise = genfromtxt(comp_noise, delimiter=None, dtype=None, skip_header=1)
    censor_vol_list = genfromtxt(vols_to_censor, delimiter=None, dtype=None, skip_header=0)
    
    try:
        c = censor_vol_list.size
    except:
        c = 0
    
    d=len(comp_noise)

    if c > 1:
        scrubbing = zeros((d,c),dtype=int)
        for t in range(c):
            scrubbing[censor_vol_list[t],t] = 1
        noise_matrix = column_stack((motion,comp_noise,scrubbing))
    elif c == 1:
        scrubbing = zeros((d,c),dtype=int)
        scrubbing[censor_vol_list] = 1
        noise_matrix = column_stack((motion,comp_noise,scrubbing))
    else:
        noise_matrix = column_stack((motion,comp_noise))
    
    noise_file = 'noise_matrix.txt'
    savetxt(noise_file, noise_matrix)
    noise_filepath = path.abspath(noise_file)
    
    return(noise_filepath)

# Artifact detection for scrubbing/motion assessment
art = Node(ArtifactDetect(mask_type='file',
                          parameter_source='FSL',
                          norm_threshold=1.0, #mutually exclusive with rotation and translation thresh
                          zintensity_threshold=3,
                          use_differences=[True, False]),
           name='art')

# Segment structural scan
segment = Node(FAST(no_bias=True, 
                    segments=True, 
                    number_classes=3), 
               name='segment')

# Fix the segmentations
fix_confs = Node(name='fix_confs',
                 interface=Function(input_names=['masks'], 
                                    output_names=['vols'],
                                    function=adjust_masks))
# actually run compcor
compcor = Node(CompCor(merge_method='none'), 
               name='compcor')

# Create a denoising mask with compcor + motion
noise_mat = Node(name='noise_mat', interface=Function(input_names=['vols_to_censor','motion_params','comp_noise'],
                                                      output_names=['noise_filepath'], 
                                                      function=create_noise_matrix))

# Denoise the data
denoise = Node(GLM(out_res_name='denoised_residuals.nii', 
                   out_data_name='denoised_func.nii'), 
               name='denoise')

# Nuissance regression -ICA AROMA:  
    # motion_parameters, in_file; aggr_denoised_file, nonaggr_denoised_file
    # --> removed 9/27/17 because the included/required masks weren't in the correct space
#ica_aroma = Node(ICA_AROMA(TR=TR,
#                           denoise_type='both'), 
#                 name='ica_aroma')


# In[ ]:

# QC nodes
def create_coreg_plot(epi,anat):
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from nipype.interfaces.nipy.preprocess import Trim
    
    epiVol = 'firstVol.nii'
    trim = Trim()
    trim.inputs.in_file = epi
    trim.inputs.out_file = epiVol
    trim.inputs.end_index = 1
    trim.inputs.begin_index = 0
    trim.run()
    
    coreg_filename='coregistration.png'
    display = plotting.plot_anat(epiVol, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'coregistration to anatomy')
    display.add_edges(anat)
    display.savefig(coreg_filename) 
    display.close()
    coreg_file = abspath(coreg_filename)
    
    return(coreg_file)

def check_mask_coverage(epi,brainmask):
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from numpy import sum, asarray, vstack
    from nipype.interfaces.nipy.preprocess import Trim
    
    epiVol = 'firstVol.nii'
    trim = Trim()
    trim.inputs.in_file = epi
    trim.inputs.out_file = epiVol
    trim.inputs.end_index = 1
    trim.inputs.begin_index = 0
    trim.run()
    
    maskcheck_filename='maskcheck.png'
    display = plotting.plot_anat(epiVol, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'check brainmask coverage')
    display.add_contours(brainmask,levels=[.5], colors='r')
    display.savefig(maskcheck_filename) 
    display.close()
    

    maskcheck_file = abspath(maskcheck_filename)

    return(maskcheck_file)

make_coreg_img = Node(Function(input_names=['epi','anat'],
                                         output_names=['coreg_file'],
                                         function=create_coreg_plot),
                      name='make_coreg_img')

make_checkmask_img = Node(Function(input_names=['epi','brainmask'],
                                         output_names=['maskcheck_file'],
                                         function=check_mask_coverage),
                          name='make_checkmask_img')


# In[ ]:

# workflow
preprocflow = Workflow(name='preprocflow')
preprocflow.connect([(infosource,structgrabber,[('subjid','subjid')]),
                     (infosource,structgrabber,[('timepoint','timepoint')]),
                     (structgrabber,fs_preproc,[('struct','T1_files')]),
                     (infosource,get_fsID,[('subjid','subjid')]),
                     (infosource,get_fsID,[('timepoint','timepoint')]),
                     (get_fsID,fs_preproc,[('fs_subjid','subject_id')]),
                     (fs_preproc, convert_anat,[('brainmask','in_file')]),
                     (convert_anat,reorient_anat,[('out_file','in_file')]),
                     (reorient_anat,segment,[('out_file','in_files')]),
                     (segment,fix_confs,[('tissue_class_files','masks')]),
                     (fix_confs,compcor,[('vols','mask_files')]),
                     (reorient_anat, binarize_anat,[('out_file','in_file')]),
                     (reorient_anat,reg_func_to_anat,[('out_file','reference')]),
                     (reorient_anat,apply_reg_to_func,[('out_file','reference')]),
                     (binarize_anat,mask_func,[('binary_file','mask_file')]),
                     (binarize_anat,art,[('binary_file','mask_file')]),
                     
                     (infosource,funcgrabber,[('subjid','subjid')]),
                     (infosource,funcgrabber,[('timepoint','timepoint')]),
                     (funcgrabber,unzip_func,[('func','in_file')]),
                     (unzip_func,reorient_func,[('out_file','in_file')]),
                     (reorient_func,realign_runs,[('out_file','in_file')]),
                     (realign_runs, slicetime,[('out_file','in_file')]),
                     (slicetime,reg_func_to_anat,[('slice_time_corrected_file','in_file')]),
                     (slicetime,apply_reg_to_func,[('slice_time_corrected_file','in_file')]),
                     (reg_func_to_anat,apply_reg_to_func,[('out_matrix_file','in_matrix_file')]),
                     (apply_reg_to_func,norm_run_intensities,[('out_file','func_files')]),
                     (norm_run_intensities,merge_func,[('new_func_list','in_files')]),
                     (merge_func,realign_merged,[('merged_file','in_file')]),
                     (realign_merged,mask_func,[('out_file','in_file')]),
                     
                     (realign_runs,merge_motion,[('par_file','motion_files')]),
                     (mask_func,merge_motion,[('out_file','merged_func')]),
                     (mask_func,art,[('out_file','realigned_files')]),
                     (merge_motion,art,[('newmotion_params','realignment_parameters')]),
                     (mask_func,compcor,[('out_file','realigned_file')]),
                     (compcor,noise_mat,[('components_file','comp_noise')]),
                     (art,noise_mat,[('outlier_files','vols_to_censor')]),
                     (merge_motion,noise_mat,[('newmotion_params','motion_params')]),
                     (noise_mat,denoise,[('noise_filepath','design')]),
                     (mask_func,denoise,[('out_file','in_file')]),
                     (denoise,bandpass,[('out_data','in_file')]),
                     
                     (realign_merged,make_coreg_img,[('out_file','epi')]),
                     (reorient_anat,make_coreg_img,[('out_file','anat')]),
                     (realign_merged,make_checkmask_img,[('out_file','epi')]),
                     (binarize_anat,make_checkmask_img,[('binary_file','brainmask')]),
                     
                     (merge_func,datasink,[('merged_file','merged_func')]),
                     (make_coreg_img,datasink,[('coreg_file','coregcheck_image')]),
                     (make_checkmask_img,datasink,[('maskcheck_file','maskcheck_image')]),
                     (mask_func, datasink,[('out_file','orig_merged_func')]),
                     (reorient_anat,datasink,[('out_file','preproc_anat')]),
                     (binarize_anat,datasink,[('binary_file','binarized_anat')]),
                     (merge_motion, datasink,[('newmotion_params','motion_params')]),
                     (noise_mat,datasink,[('noise_filepath','full_noise_mat')]),
                     (art,datasink,[('plot_files','art_plot_files')]),
                     (bandpass,datasink,[('out_file','preproc_func')])        
                    ])
preprocflow.base_dir = workflow_dir
preprocflow.write_graph(graph2use='flat')
preprocflow.run('MultiProc', plugin_args={'n_procs': 10})


# In[ ]:



