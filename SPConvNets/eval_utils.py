import numpy as np
from scipy.spatial.transform import Rotation as srot
from scipy.optimize import least_squares

def rotate_points_with_rotvec(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def rotate_pts(source, target):
    '''
    func: compute rotation between source: [N x 3], target: [N x 3]
    '''
    # pre-centering
    source = source - np.mean(source, 0, keepdims=True)
    target = target - np.mean(target, 0, keepdims=True)
    M = np.matmul(target.T, source)
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    R = np.matmul(U, Vh)
    return R

def objective_eval(params, x0, y0, x1, y1, joints, isweight=True):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1,3))
    rotvec1 = params[3:].reshape((1,3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_joint = rotate_points_with_rotvec(joints, rotvec0) - rotate_points_with_rotvec(joints, rotvec1)
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
        res_joint /= joints.shape[0]
    return np.concatenate((res0, res1, res_joint), 0).ravel()

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def rot_diff_rad(rot1, rot2):
    return np.arccos( ( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2 ) % (2*np.pi)


def scale_pts(source, target):
    '''
    func: compute scaling factor between source: [N x 3], target: [N x 3]
    '''
    pdist_s = source.reshape(source.shape[0], 1, 3) - source.reshape(1, source.shape[0], 3)
    A = np.sqrt(np.sum(pdist_s**2, 2)).reshape(-1)
    pdist_t = target.reshape(target.shape[0], 1, 3) - target.reshape(1, target.shape[0], 3)
    b = np.sqrt(np.sum(pdist_t**2, 2)).reshape(-1)
    scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    return scale


def transform_pts(source, target):
    # source: [N x 3], target: [N x 3]
    # pre-centering and compute rotation
    source_centered = source - np.mean(source, 0, keepdims=True)
    target_centered = target - np.mean(target, 0, keepdims=True)
    rotation = rotate_pts(source_centered, target_centered)

    scale = scale_pts(source_centered, target_centered)
    # R ((pc * s) + T) + T_2 --> R(pc * s) + RT + T_2
    # compute translation
    translation = np.mean(target.T-scale*np.matmul(rotation, source.T), 1)
    return rotation, scale, translation


def ransac(dataset, model_estimator, model_verifier, inlier_th, niter=10000):
    best_model = None
    best_score = -np.inf
    best_inliers = None
    for i in range(niter): # model estimator
        cur_model = model_estimator(dataset)
        cur_score, cur_inliers = model_verifier(dataset, cur_model, inlier_th)
        if cur_score > best_score:
            best_model = cur_model
            best_inliers = cur_inliers
            best_score = cur_score
    if best_model is None:
        print("zzzz None!")
    best_model = model_estimator(dataset, best_inliers)
    return best_model, best_inliers


def single_transformation_estimator(dataset, best_inliers = None):
    # dataset: dict, fields include source, target, nsource
    if best_inliers is None:
        sample_idx = np.random.randint(dataset['nsource'], size=30)
    else:
        sample_idx = best_inliers
    rotation, scale, translation = transform_pts(dataset['source'][sample_idx,:], dataset['target'][sample_idx,:])
    strans = dict()
    strans['rotation'] = rotation
    strans['scale'] = scale
    strans['translation'] = translation
    return strans


def single_transformation_verifier(dataset, model, inlier_th):
    # dataset: dict, fields include source, target, nsource, ntarget
    # model: dict, fields include rotation, scale, translation
    res = dataset['target'].T - model['scale'] * np.matmul( model['rotation'], dataset['source'].T ) - model['translation'].reshape((3, 1))
    inliers = np.sqrt(np.sum(res**2, 0)) < inlier_th
    score = np.sum(inliers)
    return score, inliers


def joint_transformation_estimator(dataset, best_inliers = None):
    # dataset: dict, fields include source0, target0, nsource0,
    #     source1, target1, nsource1, joint_direction
    if best_inliers is None:
        sample_idx0 = np.random.randint(dataset['nsource0'], size=3)
        sample_idx1 = np.random.randint(dataset['nsource1'], size=3)
    else:
        sample_idx0 = best_inliers[0]
        sample_idx1 = best_inliers[1]

    source0 = dataset['source0'][sample_idx0, :]
    target0 = dataset['target0'][sample_idx0, :]
    source1 = dataset['source1'][sample_idx1, :]
    target1 = dataset['target1'][sample_idx1, :]

    # prescaling and centering
    scale0 = scale_pts(source0, target0)
    scale1 = scale_pts(source1, target1)
    scale0_inv = scale_pts(target0, source0) # check if could simply take reciprocal
    scale1_inv = scale_pts(target1, source1)

    target0_scaled_centered = scale0_inv*target0
    target0_scaled_centered -= np.mean(target0_scaled_centered, 0, keepdims=True)
    source0_centered = source0 - np.mean(source0, 0, keepdims=True)

    target1_scaled_centered = scale1_inv*target1
    target1_scaled_centered -= np.mean(target1_scaled_centered, 0, keepdims=True)
    source1_centered = source1 - np.mean(source1, 0, keepdims=True)

    joint_points0 = np.ones_like(np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1)))*dataset['joint_direction'].reshape((1, 3))
    joint_points1 = np.ones_like(np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1)))*dataset['joint_direction'].reshape((1, 3))

    R0 = rotate_pts(source0_centered, target0_scaled_centered)
    R1 = rotate_pts(source1_centered, target1_scaled_centered)
    rdiff0 = np.inf
    rdiff1 = np.inf
    niter = 10
    degree_th = 0.05
    isalternate = False

    if not isalternate:
        rotvec0 = srot.from_dcm(R0).as_rotvec()
        rotvec1 = srot.from_dcm(R1).as_rotvec()
        res = least_squares(objective_eval, np.hstack((rotvec0, rotvec1)), verbose=0, ftol=1e-4, method='lm',
                        args=(source0_centered, target0_scaled_centered, source1_centered, target1_scaled_centered, joint_points0, False))
        R0 = srot.from_rotvec(res.x[:3]).as_dcm()
        R1 = srot.from_rotvec(res.x[3:]).as_dcm()
    else:
        for i in range(niter):
            if rdiff0<=degree_th and rdiff1<=degree_th:
                break
            newsrc0 = np.concatenate( (source0_centered, joint_points0), 0 )
            newtgt0 = np.concatenate( (target0_scaled_centered, np.matmul( joint_points0, R1.T ) ), 0 )
            newR0 = rotate_pts( newsrc0, newtgt0 )
            rdiff0 = rot_diff_degree(R0, newR0)
            R0 = newR0

            newsrc1 = np.concatenate( (source1_centered, joint_points1), 0 )
            newtgt1 = np.concatenate( (target1_scaled_centered, np.matmul( joint_points1, R0.T ) ), 0 )
            newR1 = rotate_pts( newsrc1, newtgt1 )
            rdiff1 = rot_diff_degree(R1, newR1)
            R1 = newR1

    translation0 = np.mean(target0.T-scale0*np.matmul(R0, source0.T), 1)
    translation1 = np.mean(target1.T-scale1*np.matmul(R1, source1.T), 1)
    jtrans = dict()
    jtrans['rotation0'] = R0
    jtrans['scale0'] = scale0
    jtrans['translation0'] = translation0
    jtrans['rotation1'] = R1
    jtrans['scale1'] = scale1
    jtrans['translation1'] = translation1
    return jtrans


def joint_transformation_verifier(dataset, model, inlier_th):
    # dataset: dict, fields include source, target, nsource, ntarget
    # model: dict, fields include rotation, scale, translation
    res0 = dataset['target0'].T - model['scale0'] * np.matmul( model['rotation0'], dataset['source0'].T ) - model['translation0'].reshape((3, 1))
    inliers0 = np.sqrt(np.sum(res0**2, 0)) < inlier_th
    res1 = dataset['target1'].T - model['scale1'] * np.matmul( model['rotation1'], dataset['source1'].T ) - model['translation1'].reshape((3, 1))
    inliers1 = np.sqrt(np.sum(res1**2, 0)) < inlier_th
    score = ( np.sum(inliers0)/res0.shape[0] + np.sum(inliers1)/res1.shape[0] ) / 2
    return score, [inliers0, inliers1]

