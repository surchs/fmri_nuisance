import numpy as np
import pandas as pd
import pathlib as pal
import nibabel as nib
import argparse as arg
from sklearn import linear_model as sln
from sklearn import decomposition as dp
from sklearn import preprocessing as skp
from nipype.algorithms import confounds as nac


def build_regressors(reg_p, fd_p, csf_p, wm_p):
    # Get the regressors
    movement_regressors = pd.read_csv(reg_p, delim_whitespace=True, header=None)
    headers = ['{}{}'.format(i, j) for i in ['t', 'r', 'd_t', 'd_r'] for j in
               ['x', 'y', 'z']]
    movement_regressors.rename(columns={key: headers[key] for key in range(12)},
                               inplace=True)

    fd = pd.read_csv(fd_p, delim_whitespace=True, header=None)
    fd.rename(columns={0: 'fd'}, inplace=True)
    motion = movement_regressors.join(fd, how='outer')
    motion_columns = ['{}{}'.format(i, j) for i in ['t', 'r'] for j in
                      ['x', 'y', 'z']]
    motion_linear = motion[motion_columns]
    # Get their squares as well
    motion_columns_square = ['{}_sq'.format(name) for name in motion_columns]
    motion_square = pd.DataFrame(
        data=np.array([np.square(motion_linear[name].values)
                       for name in motion_columns]).T,
        columns=motion_columns_square)
    motion_input = motion_linear.join(motion_square, how='outer')
    # Run the PCA
    # Standardize the input
    scale = skp.StandardScaler()
    stand = scale.fit_transform(motion_input)
    # Get the top 95% variance components
    pca = dp.PCA(n_components=0.95)
    X_comp = pca.fit_transform(stand)
    motion_comp = pd.DataFrame(data=X_comp,
                               columns=['pca_{}'.format(i + 1) for i in
                                        range(X_comp.shape[1])])

    csf_signal = pd.read_csv(csf_p, delim_whitespace=True, header=None)
    csf_signal.rename(columns={0: 'csf'}, inplace=True)
    wm_signal = pd.read_csv(wm_p, delim_whitespace=True, header=None)
    wm_signal.rename(columns={0: 'wm'}, inplace=True)
    tissue_signal = csf_signal.join(wm_signal, how='outer')
    # Join everything together
    nuisance = motion_comp.join(tissue_signal, how='outer')

    return nuisance


def build_DCT(data, TR=1, longest_cycle_time=1/0.008):
    reg, base = nac.cosine_filter(data=data, timestep=TR,
                                  period_cut=longest_cycle_time)
    # Store cosine basis
    dc_base = pd.DataFrame(data=base, columns=['dc_{}'.format(i + 1) for i in
                                               range(base.shape[1])])

    return dc_base


def build_motion_mask(fd_p, threshold=0.2):
    fd = pd.read_csv(fd_p, delim_whitespace=True, header=None)
    fd.rename(columns={0: 'fd'}, inplace=True)
    n_t = fd.shape[0]
    motion_mask = (fd.fd > threshold).values
    motion_ind = np.argwhere(motion_mask).flatten()
    scrub_mask = np.zeros(n_t, dtype=bool)

    # Build the censoring mask
    for i in motion_ind:
        if i == 0:
            scrub_mask[i:i + 2] = True
        elif i > n_t - 3:
            scrub_mask[i-1:] = True
        else:
            scrub_mask[i-1:i + 2] = True

    return scrub_mask


def main(base_p, mask_p, sub_name):
    # Make sure we have pathlib objects
    if not type(base_p):
        base_p = pal.Path(base_p)
    # Get the folder of this guy but only for the FMRI data
    sub_p = base_p / sub_name / 'FMRI'
    # Get the list of functional folders for this subject
    func_folders = [f for f in sub_p.glob('rfMRI_REST*')]

    # Get the mask
    mask = nib.load(str(mask_p)).get_data().astype(bool)
    # Go through these folders
    for func in func_folders:
        # Prepare the paths for the in and outputs
        in_img_p = func / (func.name + '.nii.gz')
        out_img_p = func / (func.name + '_residuals.nii.gz')
        reg_p = func / 'Movement_Regressors.txt'
        fd_p = func / 'Movement_RelativeRMS.txt'
        csf_p = func / (func.name + '_CSF.txt')
        wm_p = func / (func.name + '_WM.txt')

        # Load the data
        img = nib.load(str(in_img_p))
        data_vec = img.get_data()[mask, :]
        n_t = data_vec.shape[1]
        # Build the regression model
        nuisance = build_regressors(reg_p, fd_p, csf_p, wm_p)
        dct_base = build_DCT(data_vec, TR=0.72, longest_cycle_time=1/0.008)
        regressors = nuisance.join(dct_base, how='outer')
        scrub_mask = build_motion_mask(fd_p, threshold=0.2)

        # Fit the GLM only on the scrubbed time points
        reg_scrub = regressors.iloc[~scrub_mask, :]
        # Standardize the input
        scale = skp.StandardScaler()
        scale.fit(reg_scrub)
        reg_scrub_stand = scale.transform(reg_scrub)
        reg_stand = scale.transform(regressors)
        X_scrub = data_vec[:, ~scrub_mask].T
        X = data_vec.T

        glm = sln.LinearRegression(fit_intercept=True, n_jobs=-1)
        res = glm.fit(reg_scrub_stand, X_scrub)
        residuals = X - res.predict(reg_stand) + res.intercept_

        # Map the results back into volume
        res_vol = np.zeros(mask.shape + (n_t,), dtype='<f4')
        res_vol[mask, :] = residuals.T
        res_img = nib.Nifti1Image(res_vol, affine=img.affine, header=img.header)
        nib.save(res_img, out_img_p)


if __name__ == "__main__":
    parser = arg.ArgumentParser(description='Regress nuisance covariates from '
                                            'HCP FMRI data')
    parser.add_argument('hcp_p', help='full path to the HCP subjects folder',
                        type=pal.Path)
    parser.add_argument('mask_p', help='full path to the brainmask used for '
                                       'the HCP subjects', type=pal.Path)
    parser.add_argument('subname', help='the name or ID of the '
                                        'subject to be run', type=str)

    args = parser.parse_args()
    main(args.hcp_p, args.mask_p, args.subname)