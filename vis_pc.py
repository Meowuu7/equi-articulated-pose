# import open3d as o3d
import numpy as np
import os
# from open3d import *

import polyscope as ps

from sklearn.manifold import TSNE

ps.init()
ps.set_ground_plane_mode("none")
ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
ps.set_screenshot_extension(".png")

color = [
    (136/255.0,224/255.0,239/255.0),
    (180/255.0,254/255.0,152/255.0),
    (184/255.0,59/255.0,94/255.0),
    (106/255.0,44/255.0,112/255.0),
    (39/255.0,53/255.0,135/255.0),
(0,173/255.0,181/255.0), (170/255.0,150/255.0,218/255.0), (82/255.0,18/255.0,98/255.0), (234/255.0,84/255.0,85/255.0), (234/255.0,255/255.0,208/255.0),(162/255.0,210/255.0,255/255.0),
    (187/255.0,225/255.0,250/255.0), (240/255.0,138/255.0,93/255.0), (184/255.0,59/255.0,94/255.0),(106/255.0,44/255.0,112/255.0),(39/255.0,53/255.0,135/255.0),
]

# anchors = np.load(os.path.join("data", "kernels.npy"), allow_pickle=True)
# print("kernels.shaep = ", anchors.shape)
#
# cur_pcd = ps.register_point_cloud(f"kernels", anchors, radius=0.012, color=color)
#
# ps.show()
# ps.remove_all_structures()

# pc = PointCloud()
# pc.points = Vector3dVector(anchors)
# draw_geometries([pc])

''' Some basic data: features, points, labels '''
# labels = np.load("./data/vis_labels.npy", allow_pickle=True)
# pts = np.load("./data/vis_pts.npy", allow_pickle=True)
# slot_pts = np.load("./data/slot_pts.npy", allow_pickle=True)
# recon_pts = np.load("./data/downsampled_pts.npy", allow_pickle=True)
# ori_pts = np.load("./data/ori_pts.npy", allow_pickle=True)
#
# bz = labels.shape[0]


# for j in range(bz):
#     cur_label = labels[j]
#     cur_pts = pts[j]
#     seg_label_to_pts = {}
#     for i in range(cur_label.shape[0]):
#         curr_lab = int(cur_label[i].item())
#         if curr_lab not in seg_label_to_pts:
#             seg_label_to_pts[curr_lab] = [i]
#         else:
#             seg_label_to_pts[curr_lab].append(i)
#     for seg_label in seg_label_to_pts:
#         cur_pts_idx = np.array(seg_label_to_pts[seg_label])
#         cur_pts_pts = cur_pts[:, cur_pts_idx]
#         cur_pts_pts = np.transpose(cur_pts_pts, (1, 0))
#         cur_color = color[seg_label]
#         cur_pcd = ps.register_point_cloud(f"seg_{seg_label}", cur_pts_pts, radius=0.012, color=cur_color)
#     ps.show()
#     ps.remove_all_structures()
# #
# for j in range(bz):
#     for j_slot in range(slot_pts.shape[1]):
#         cur_color = color[j_slot]
#         cur_pcd = ps.register_point_cloud(f"seg_{j_slot}", slot_pts[j,j_slot], radius=0.012, color=cur_color)
#     ps.show()
#     ps.remove_all_structures()

# if recon_pts.shape[1] == 3:
#     recon_pts = np.transpose(recon_pts, (2, 1))
# if ori_pts.shape[1] == 3:
#     print(ori_pts.shape)
#     ori_pts = np.transpose(ori_pts, (0, 2, 1))
# bz = ori_pts.shape[0]
#
# print(f"ori_pts.shape: {ori_pts.shape}, recon_pts.shape: {recon_pts.shape}")
#
# for j in range(bz):
#     ori_color = color[1]
#     cur_color = color[0]
#     cur_pcd = ps.register_point_cloud(f"whole", recon_pts[j], radius=0.012, color=cur_color)
#     cur_pcd_ori = ps.register_point_cloud(f"ori", ori_pts[j], radius=0.012, color=ori_color)
#     print(ori_pts[j, 0:10,:])
#     # j
#     ps.show()
#     ps.remove_all_structures()


# what can we do if we want the network to reconstruct per-part points?

def vis_predicted_slots_shapes():

    ''' Plot points for each slot '''
    ''' With slot reconstruction '''
    # slot_recon_pts = np.load(os.path.join("data", "recon_slot_pts.npy"), allow_pickle=True)
    # sampled_recon_pts = np.load(os.path.join("data", "sampled_recon_pts.npy"), allow_pickle=True)
    # labels_ori_pts = np.load(os.path.join("data", "vis_labels.npy"), allow_pickle=True)
    # ori_pts = np.load(os.path.join("data", "vis_pts.npy"), allow_pickle=True)

    ''' Not a single shape, a shape category '''
    ''' With slot reconstruction '''
    # slot_recon_pts = np.load(os.path.join("data", "recon_slot_pts_multi.npy"), allow_pickle=True)
    # sampled_recon_pts = np.load(os.path.join("data", "sampled_recon_pts_multi.npy"), allow_pickle=True)
    # labels_ori_pts = np.load(os.path.join("data", "vis_labels_multi.npy"), allow_pickle=True)
    # ori_pts = np.load(os.path.join("data", "vis_pts_multi.npy"), allow_pickle=True)
    # #

    ''' Plot points for each slot '''
    ''' With slot reconstruction '''
    # slot_recon_pts = np.load(os.path.join("data", "recon_slot_pts_hard.npy"), allow_pickle=True)
    # sampled_recon_pts = np.load(os.path.join("data", "sampled_recon_pts_hard.npy"), allow_pickle=True)
    # labels_ori_pts = np.load(os.path.join("data", "vis_labels_hard.npy"), allow_pickle=True)
    # ori_pts = np.load(os.path.join("data", "vis_pts_hard.npy"), allow_pickle=True)
    #
    # ''' Plot points for each slot '''
    # ''' With slot reconstruction '''
    # slot_recon_pts = np.load(os.path.join("data", "recon_slot_pts_hard_2.npy"), allow_pickle=True)
    # sampled_recon_pts = np.load(os.path.join("data", "sampled_recon_pts_hard_2.npy"), allow_pickle=True)
    # labels_ori_pts = np.load(os.path.join("data", "vis_labels_hard_2.npy"), allow_pickle=True)
    # ori_pts = np.load(os.path.join("data", "vis_pts_hard_2.npy"), allow_pickle=True)
    #
    #
    ''' Plot points for each slot '''
    ''' With slot reconstruction '''
    # slot_recon_pts = np.load(os.path.join("data", "recon_slot_pts_hard_3.npy"), allow_pickle=True)
    # sampled_recon_pts = np.load(os.path.join("data", "sampled_recon_pts_hard_3.npy"), allow_pickle=True)
    # labels_ori_pts = np.load(os.path.join("data", "vis_labels_hard_3.npy"), allow_pickle=True)
    # ori_pts = np.load(os.path.join("data", "vis_pts_hard_3.npy"), allow_pickle=True)

    ''' Plot points for each slot '''
    ''' With slot reconstruction '''
    # slot_recon_pts = np.load(os.path.join("data", "recon_slot_pts_hard_4.npy"), allow_pickle=True)
    # sampled_recon_pts = np.load(os.path.join("data", "sampled_recon_pts_hard_4.npy"), allow_pickle=True)
    # labels_ori_pts = np.load(os.path.join("data", "vis_labels_hard_4.npy"), allow_pickle=True)
    # ori_pts = np.load(os.path.join("data", "vis_pts_hard_4.npy"), allow_pickle=True)

    # out_feats = np.load(os.path.join("data", "out_feats_6.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_5.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_12.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_13_w_r.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_13_w_r_wp.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_13_wo_r_wp.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_11.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_14_wo_r_no_equi.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_14_w_r_no_equi_lbz.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_with_features_14_w_r_no_equi_lbz.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_0_num_iters_1_npts_256.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_0_num_iters_1_npts_400.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_woeq_wrot_0_num_iters_1_npts_400_bsz_2_init_lr_1e-05.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_woeq_wrot_0_num_iters_1_npts_512_perpart_npts_512_bsz_16_init_lr_1e-05.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_0_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_1e-05.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_0_num_iters_1_npts_400_bsz_2_init_lr_1e-05.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_num_iters_1_npts_400_perpart_npts_128_bsz_2_init_lr_1e-05.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_num_iters_1_npts_400_perpart_npts_128_bsz_2_init_lr_1e-05.npy"), allow_pickle=True).item()
    out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_equi_2_num_iters_1_npts_512_perpart_npts_256_bsz_1_init_lr_0.0001.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_equi_2_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_5e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_equi_2_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_1e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_equi_2_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_0.0001.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_equi_1_num_iters_1_npts_400_perpart_npts_128_bsz_2_init_lr_0.0001.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_equi_1_num_iters_1_npts_400_perpart_npts_128_bsz_2_init_lr_5e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_equi_2_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_5e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_5e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_0_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_1e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_5e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_num_iters_1_npts_512_perpart_npts_128_bsz_1_init_lr_5e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_woeq_wrot_0_num_iters_1_npts_512_perpart_npts_512_bsz_16_init_lr_1e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_1_num_iters_1_npts_400_perpart_npts_128_bsz_2_init_lr_1e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_weq_wrot_0_num_iters_2_npts_390_bsz_2_init_lr_1e-05.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_14_w_r_no_equi.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "chair_out_feats_7.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_8.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_5.npy"), allow_pickle=True).item()
    # out_feats = np.load(os.path.join("data", "out_feats_with_features_6.npy"), allow_pickle=True).item()
    if out_feats["ori_recon_slot_pts_hard"].shape[1] == 1:
        ori_slot_recon_pts = out_feats["ori_recon_slot_pts_hard"][:, 0, ...]
        slot_recon_pts = out_feats["recon_slot_pts_hard"][:, 0, ...]
        sampled_recon_pts = out_feats["sampled_recon_pts_hard"][:, 0, ...]
        labels_ori_pts = out_feats["vis_labels_hard"][:, 0, ...]
        ori_pts = out_feats["vis_pts_hard"]
    else:
        ori_slot_recon_pts = out_feats["ori_recon_slot_pts_hard"]
        slot_recon_pts = out_feats["recon_slot_pts_hard"]
        sampled_recon_pts = out_feats["sampled_recon_pts_hard"]
        labels_ori_pts = out_feats["vis_labels_hard"]
        ori_pts = out_feats["vis_pts_hard"]

    print(f"ori_slot_recon_pts: {ori_slot_recon_pts.shape}, slot_recon_pts: {slot_recon_pts.shape}, sampled_recon_pts: {sampled_recon_pts.shape}, labels_ori_pts: {labels_ori_pts.shape}, ori_pts: {ori_pts.shape}")

    ''' Plot points for each slot '''
    ''' With slot reconstruction '''
    # slot_recon_pts = np.load(os.path.join("data", "recon_slot_pts_hard_5.npy"), allow_pickle=True)
    # sampled_recon_pts = np.load(os.path.join("data", "sampled_recon_pts_hard_5.npy"), allow_pickle=True)
    # labels_ori_pts = np.load(os.path.join("data", "vis_labels_hard_5.npy"), allow_pickle=True)
    # ori_pts = np.load(os.path.join("data", "vis_pts_hard_5.npy"), allow_pickle=True)

    if ori_pts.shape[1] == 3:
        ori_pts = np.transpose(ori_pts, (0, 2, 1))
    # print(f"slot reconstruction points : {slot_recon_pts.shape}, sampled recon points : {sampled_recon_pts.shape}")

    for i in range(0, slot_recon_pts.shape[0]):
        ''' Plot point clouds from different slots '''
        cur_slot_recon_pts = slot_recon_pts[i]
        for i_slot in range(cur_slot_recon_pts.shape[0]):
            cur_color = color[i_slot]
            cur_slot_pts = cur_slot_recon_pts[i_slot]
            cur_pcd = ps.register_point_cloud(f"seg_{i_slot}", cur_slot_pts, radius=0.012, color=cur_color)
        ps.show()
        ps.remove_all_structures()

        ''' Plot ori point clouds from different slots '''
        cur_ori_slot_recon_pts = ori_slot_recon_pts[i]
        for i_slot in range(cur_ori_slot_recon_pts.shape[0]):
            cur_color = color[i_slot]
            cur_slot_pts = cur_ori_slot_recon_pts[i_slot]
            cur_pcd = ps.register_point_cloud(f"seg_ori_recon_{i_slot}", cur_slot_pts, radius=0.012, color=cur_color)
        ps.show()
        ps.remove_all_structures()

        ''' Plot the reconstructed point cloud '''
        cur_sampled_recon_pts = sampled_recon_pts[i]
        cur_pcd = ps.register_point_cloud(f"tot", cur_sampled_recon_pts, radius=0.012, color=color[0])
        ps.show()
        ps.remove_all_structures()

        ''' Plot the segmented point cloud '''
        cur_ori_pts = ori_pts[i]
        cur_labels_ori_pts = labels_ori_pts[i]
        seg_idx_to_pts_idx = {}
        for j in range(cur_labels_ori_pts.shape[0]):
            cur_pts_label = int(cur_labels_ori_pts[j].item())
            if cur_pts_label not in seg_idx_to_pts_idx:
                seg_idx_to_pts_idx[cur_pts_label] = [j]
            else:
                seg_idx_to_pts_idx[cur_pts_label].append(j)
        for seg_idx in seg_idx_to_pts_idx:
            cur_seg_pts_idxes = np.array(seg_idx_to_pts_idx[seg_idx], dtype=np.long)
            cur_seg_pts = cur_ori_pts[cur_seg_pts_idxes]
            cur_color = color[seg_idx]
            cur_pcd = ps.register_point_cloud(f"seg_{seg_idx}", cur_seg_pts, radius=0.012, color=cur_color)
        ps.show()
        ps.remove_all_structures()


''' Plot sampled reconstruction points '''
# for i in range(sampled_recon_pts.shape[0]):
#     cur_sampled_recon_pts = sampled_recon_pts[i]
#     cur_pcd = ps.register_point_cloud(f"tot", cur_sampled_recon_pts, radius=0.012, color=color[0])
#     ps.show()
#     ps.remove_all_structures()


''' Only whole reconstruction '''
# ori_pts = np.load(os.path.join("data", "ori_pts.npy"), allow_pickle=True)
# recon_pts = np.load(os.path.join("data", "downsampled_pts.npy"), allow_pickle=True)
#
# if ori_pts.shape[1] == 3:
#     ori_pts = np.transpose(ori_pts, (0, 2, 1))
#
# for i in range(ori_pts.shape[0]):
#     cur_ori_pts = ori_pts[i]
#     cur_recon_pts = recon_pts[i]
#     cur_pcd = ps.register_point_cloud(f"ori", cur_ori_pts, radius=0.012, color=color[0])
#     recon_pcd = ps.register_point_cloud(f"recon", cur_recon_pts, radius=0.012, color=color[1])
#     ps.show()
#     ps.remove_all_structures()

def plot_features():
    ''' dummy '''
    ''' Plot the feature w.r.t. clustering patterns? '''
    x_features_path = os.path.join("data", "x_features_hard_4.npy")
    # x_features_path = os.path.join("/home/xueyi/EPN_PointCloud", "x_features.npy") #
    labels_ori_pts = np.load(os.path.join("data", "vis_labels_hard_4.npy"), allow_pickle=True)

    x_features = np.load(x_features_path, allow_pickle=True)

    tot_feats = np.load(os.path.join("data", "out_feats_with_features_6.npy"), allow_pickle=True).item()
    x_features = tot_feats["x_features_hard"]
    ori_pts = tot_feats["vis_pts_hard"]

    print(f"x_features.shape: {x_features.shape}")
    print(f"labels_ori_pts.shape: {labels_ori_pts.shape}")

    # ori_pts = np.load(os.path.join("data", "vis_pts_hard_4.npy"), allow_pickle=True)

    if ori_pts.shape[1] == 3:
        ori_pts = np.transpose(ori_pts, (0, 2, 1))

    x_features = np.transpose(x_features, (0, 2, 1))

    bz, feat_dim, N = x_features.shape[0], x_features.shape[1], x_features.shape[2]
    print(f"batch size: {bz}, feature dimension: {feat_dim}, number of points: {N}.")

    for i in range(0, bz):
        cur_bz_x_feature = x_features[i]
        cur_bz_x_labels = labels_ori_pts[i]
        cur_ori_pts = ori_pts[i]
        cur_labels_ori_pts = labels_ori_pts[i]
        cur_bz_x_feature = cur_bz_x_feature.astype('float')

        print(type(cur_bz_x_feature))
        X_embedded = TSNE(n_components=2, init='random').fit_transform(cur_bz_x_feature)

        print(f"X_embedded.shape: {X_embedded.shape}")

        from plotly.offline import init_notebook_mode, iplot, plot
        import plotly.graph_objs as go

        x_vals, y_vals = X_embedded[:, 0], X_embedded[:, 1]
        x_vals, y_vals = x_vals.tolist(), y_vals.tolist()

        down_dim_labels = [0 if (xx < 0 and yy < 0) else 1 if (xx < 0 and yy >= 0) else 2 if (xx >= 0 and yy < 0) else 3 for (xx, yy) in zip(x_vals, y_vals)]

        cur_bz_pts_labels = cur_bz_x_labels.tolist()
        cur_bz_pts_labels = [str(lab) for lab in cur_bz_pts_labels]

        cur_bz_pts_corr_labels = []
        for jj in range(cur_ori_pts.shape[0]):
            curr_pts_xyz = cur_ori_pts[jj].tolist()
            curr_pts_xyz = ["{:.2f}".format(xx) for xx in curr_pts_xyz][:1]
            curr_pts_xyz = ",".join(curr_pts_xyz)
            cur_bz_pts_corr_labels.append(curr_pts_xyz)

        # trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=cur_bz_pts_labels)
        trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=cur_bz_pts_corr_labels)
        data = [trace]

        plot(data, filename='word-embedding-plot-two-dim-scattering.html')

        ''' Plot the segmented point cloud '''

        cur_labels_ori_pts = np.array(down_dim_labels)
        seg_idx_to_pts_idx = {}
        for j in range(cur_labels_ori_pts.shape[0]):
            cur_pts_label = int(cur_labels_ori_pts[j].item())
            if cur_pts_label not in seg_idx_to_pts_idx:
                seg_idx_to_pts_idx[cur_pts_label] = [j]
            else:
                seg_idx_to_pts_idx[cur_pts_label].append(j)


        for seg_idx in seg_idx_to_pts_idx:
            cur_seg_pts_idxes = np.array(seg_idx_to_pts_idx[seg_idx], dtype=np.long)
            cur_seg_pts = cur_ori_pts[cur_seg_pts_idxes]
            cur_color = color[seg_idx]
            cur_pcd = ps.register_point_cloud(f"seg_{seg_idx}", cur_seg_pts, radius=0.012, color=cur_color)
        ps.show()
        ps.remove_all_structures()

        break

#
def vis_original_reconstructed_shapes():
    ''' dummy '''
    ''' Only vis original pts and reconstructed pts '''
    recon_pts = np.load(os.path.join("data", "reconstructed_pts_only_recon.npy"), allow_pickle=True)
    ori_pts = np.load(os.path.join("data", "ori_pts_only_recon.npy"), allow_pickle=True)

    if recon_pts.shape[1] == 3:
        recon_pts = np.transpose(recon_pts, (0, 2, 1))
    if ori_pts.shape[1] == 3:
        ori_pts = np.transpose(ori_pts, (0, 2, 1))
    #
    bz = recon_pts.shape[0]

    for i in range(bz):
        cur_recon_pts = recon_pts[i]
        cur_ori_pts = ori_pts[i]
        cur_pcd = ps.register_point_cloud(f"ori", cur_ori_pts, radius=0.012, color=color[0])
        recon_pcd = ps.register_point_cloud(f"recon", cur_recon_pts, radius=0.012, color=color[1])

        ps.show()
        ps.remove_all_structures()


def vis_chair_shapes():
    ''' dummy '''
    ''' Vis Chair shapes '''
    shapes = np.load(os.path.join("data", "motion_part_meta_info_merged.npy"), allow_pickle=True).item()
    print(len(shapes))
    for shp_idx in shapes:
        print(shp_idx)
        cur_shp_pts = shapes[shp_idx]['pc1']
        recon_pcd = ps.register_point_cloud(f"recon", cur_shp_pts, radius=0.012, color=color[0])
        ps.show()
        ps.remove_all_structures()


if __name__=='__main__':
    vis_predicted_slots_shapes()

