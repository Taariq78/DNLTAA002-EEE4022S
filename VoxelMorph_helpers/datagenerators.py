''' The core components of the following code was provided by -

VoxelMorph: A Learning Framework for Deformable Medical Image Registration
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231 -

and was modified to better suit the project's needs'''

import numpy as np
import sys

def load_volfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data'
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % datafile

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nib' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()

    else:  # npz
        X = np.load(datafile)['vol_data']

    return X

def load_segfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data'
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % datafile

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nib' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()

    else:  # npz
        X_seg = np.load(datafile)['seg']   

    return X_seg


def example_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """
    # shuffle training data
    rng=np.random.default_rng()
    idxes = rng.choice(len(vol_names), size=len(vol_names), replace=False)
    while True:
        X_data = []
        for idx in range(batch_size):
            X = load_volfile(vol_names[idxes[idx]])
            
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # also return segmentations
        if return_segs:
            X_data = []
            temp=idxes
            for idx in range(batch_size):
                X_seg = load_segfile(vol_names[idxes[idx]])

                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)

                temp=temp[1:]
            idxes=temp
            
            # Re-shuffle data once entire set has been iterated through
            if len(idxes)==0:
              idxes = rng.choice(len(vol_names), size=len(vol_names), replace=False)

            if batch_size > 1:
                return_vals_seg = [np.concatenate(X_data, 0)]
            else:
                return_vals_seg.append(X_data[0])

        yield return_vals, return_vals_seg 
