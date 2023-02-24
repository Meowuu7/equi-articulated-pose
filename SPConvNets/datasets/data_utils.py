import numpy as np
import math
import json
import torch
import open3d as o3d

DRAWER_COMBINATIONS = [(0.1, 0.4, 0.6000000000000001), (0.1, 0.4, 0.7000000000000001), (0.1, 0.4, 0.8), (0.1, 0.4, 0.9), (0.1, 0.4, 1.0), (0.1, 0.5, 0.7000000000000001), (0.1, 0.5, 0.8), (0.1, 0.5, 0.9), (0.1, 0.5, 1.0), (0.1, 0.6000000000000001, 0.4), (0.1, 0.6000000000000001, 0.9), (0.1, 0.6000000000000001, 1.0), (0.1, 0.7000000000000001, 0.4), (0.1, 0.7000000000000001, 0.5), (0.1, 0.7000000000000001, 1.0), (0.1, 0.8, 0.4), (0.1, 0.8, 0.5), (0.1, 0.9, 0.4), (0.1, 0.9, 0.5), (0.1, 0.9, 0.6000000000000001), (0.1, 1.0, 0.4), (0.1, 1.0, 0.5), (0.1, 1.0, 0.6000000000000001), (0.1, 1.0, 0.7000000000000001), (0.2, 0.5, 0.7000000000000001), (0.2, 0.5, 0.8), (0.2, 0.5, 0.9), (0.2, 0.5, 1.0), (0.2, 0.6000000000000001, 0.9), (0.2, 0.6000000000000001, 1.0), (0.2, 0.7000000000000001, 0.5), (0.2, 0.7000000000000001, 1.0), (0.2, 0.8, 0.5), (0.2, 0.9, 0.5), (0.2, 0.9, 0.6000000000000001), (0.2, 1.0, 0.5), (0.2, 1.0, 0.6000000000000001), (0.2, 1.0, 0.7000000000000001), (0.30000000000000004, 0.6000000000000001, 0.9), (0.30000000000000004, 0.6000000000000001, 1.0), (0.30000000000000004, 0.7000000000000001, 1.0), (0.30000000000000004, 0.9, 0.6000000000000001), (0.30000000000000004, 1.0, 0.6000000000000001), (0.30000000000000004, 1.0, 0.7000000000000001), (0.4, 0.1, 0.7000000000000001), (0.4, 0.1, 0.8), (0.4, 0.1, 0.9), (0.4, 0.1, 1.0), (0.4, 0.7000000000000001, 0.1), (0.4, 0.7000000000000001, 1.0), (0.4, 0.8, 0.1), (0.4, 0.9, 0.1), (0.4, 1.0, 0.1), (0.4, 1.0, 0.7000000000000001), (0.5, 0.1, 0.8), (0.5, 0.1, 0.9), (0.5, 0.1, 1.0), (0.5, 0.2, 0.8), (0.5, 0.2, 0.9), (0.5, 0.2, 1.0), (0.5, 0.8, 0.1), (0.5, 0.8, 0.2), (0.5, 0.9, 0.1), (0.5, 0.9, 0.2), (0.5, 1.0, 0.1), (0.5, 1.0, 0.2), (0.6000000000000001, 0.1, 0.30000000000000004), (0.6000000000000001, 0.1, 1.0), (0.6000000000000001, 0.2, 1.0), (0.6000000000000001, 0.30000000000000004, 0.1), (0.6000000000000001, 0.30000000000000004, 1.0), (0.6000000000000001, 1.0, 0.1), (0.6000000000000001, 1.0, 0.2), (0.6000000000000001, 1.0, 0.30000000000000004), (0.7000000000000001, 0.1, 0.30000000000000004), (0.7000000000000001, 0.1, 0.4), (0.7000000000000001, 0.2, 0.4), (0.7000000000000001, 0.30000000000000004, 0.1), (0.7000000000000001, 0.4, 0.1), (0.7000000000000001, 0.4, 0.2), (0.8, 0.1, 0.30000000000000004), (0.8, 0.1, 0.4), (0.8, 0.1, 0.5), (0.8, 0.2, 0.4), (0.8, 0.2, 0.5), (0.8, 0.30000000000000004, 0.1), (0.8, 0.4, 0.1), (0.8, 0.4, 0.2), (0.8, 0.5, 0.1), (0.8, 0.5, 0.2), (0.9, 0.1, 0.30000000000000004), (0.9, 0.1, 0.4), (0.9, 0.1, 0.5), (0.9, 0.2, 0.4), (0.9, 0.2, 0.5), (0.9, 0.30000000000000004, 0.1), (0.9, 0.4, 0.1), (0.9, 0.4, 0.2), (0.9, 0.5, 0.1), (0.9, 0.5, 0.2)]

def get_triangles_points(vertices, triangles, selected_triangles):
    selected_triangles = triangles[selected_triangles]
    selected_triangles_vertices = []
    for i_tri in range(selected_triangles.shape[0]):
        cur_tri_idxes = selected_triangles[i_tri].tolist()
        v_a, v_b, v_c = cur_tri_idxes
        v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
        selected_triangles_vertices.append(v_a)
        selected_triangles_vertices.append(v_b)
        selected_triangles_vertices.append(v_c)
    selected_triangles_vertices = np.array(selected_triangles_vertices)
    return selected_triangles_vertices

def sample_pts_from_mesh(vertices, triangles, triangles_to_seg_idx, npoints=512):
    # arears = []
    # for i in range(triangles.shape[0]):
    #     v_a, v_b, v_c = int(triangles[i, 0].item()), int(triangles[i, 1].item()), int(
    #         triangles[i, 2].item())
    #     v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
    #     ab, ac = v_b - v_a, v_c - v_a
    #     cos_ab_ac = (np.sum(ab * ac) / np.clip(np.sqrt(np.sum(ab ** 2)) * np.sqrt(np.sum(ac ** 2)), a_min=1e-9, a_max=9999999.)).item()
    #     sin_ab_ac = math.sqrt(min(max(0., 1. - cos_ab_ac ** 2), 1.))
    #     cur_area = 0.5 * sin_ab_ac * np.sqrt(np.sum(ab ** 2)).item() * np.sqrt(np.sum(ac ** 2)).item()
    #     arears.append(cur_area)
    # tot_area = sum(arears)

    sampled_pcts = []
    pts_to_seg_idx = []
    for i in range(triangles.shape[0]):

        v_a, v_b, v_c = int(triangles[i, 0].item()), int(triangles[i, 1].item()), int(
            triangles[i, 2].item())
        v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
        ab, ac = v_b - v_a, v_c - v_a
        cos_ab_ac = (np.sum(ab * ac) / np.clip(np.sqrt(np.sum(ab ** 2)) * np.sqrt(np.sum(ac ** 2)), a_min=1e-9,
                                               a_max=9999999.)).item()
        sin_ab_ac = math.sqrt(min(max(0., 1. - cos_ab_ac ** 2), 1.))
        cur_area = 0.5 * sin_ab_ac * np.sqrt(np.sum(ab ** 2)).item() * np.sqrt(np.sum(ac ** 2)).item()

        # v_a, v_b, v_c = int(triangles[i, 0].item()), int(triangles[i, 1].item()), int(
        #     triangles[i, 2].item())
        # v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
        # ab, ac = v_b - v_a, v_c - v_a
        cur_tri_seg = int(triangles_to_seg_idx[i].item())

        # cur_sampled_pts = int(npoints * (arears[i] / tot_area))
        # cur_sampled_pts = math.ceil(npoints * (arears[i] / tot_area))
        cur_sampled_pts = int(cur_area * 500)
        cur_sampled_pts = 1 if cur_sampled_pts == 0 else cur_sampled_pts
        # if cur_sampled_pts == 0:

        tmp_x, tmp_y = np.random.uniform(0, 1., (cur_sampled_pts,)).tolist(), np.random.uniform(0., 1., (
        cur_sampled_pts,)).tolist()

        for xx, yy in zip(tmp_x, tmp_y):
            sqrt_xx, sqrt_yy = math.sqrt(xx), math.sqrt(yy)
            aa = 1. - sqrt_xx
            bb = sqrt_xx * (1. - yy)
            cc = yy * sqrt_xx
            cur_pos = v_a * aa + v_b * bb + v_c * cc
            sampled_pcts.append(cur_pos)
            pts_to_seg_idx.append(cur_tri_seg)

        # sampled_pcts.append(v_a)
        # sampled_pcts.append(v_b)
        # sampled_pcts.append(v_c)
        # pts_to_seg_idx.append(cur_tri_seg)
        # pts_to_seg_idx.append(cur_tri_seg)
        # pts_to_seg_idx.append(cur_tri_seg)
    seg_idx_to_sampled_pts = {}
    sampled_pcts = np.array(sampled_pcts, dtype=np.float)
    pts_to_seg_idx = np.array(pts_to_seg_idx, dtype=np.long)
    for i_pts in range(pts_to_seg_idx.shape[0]):
        cur_pts_seg_idx = int(pts_to_seg_idx[i_pts].item())
        if cur_pts_seg_idx not in seg_idx_to_sampled_pts:
            seg_idx_to_sampled_pts[cur_pts_seg_idx] = [i_pts]
        else:
            seg_idx_to_sampled_pts[cur_pts_seg_idx].append(i_pts)
    return sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts


def sample_pts_from_mesh_v2(vertices, triangles, pts_per_area=500):

    sampled_pcts = []
    for i in range(triangles.shape[0]):

        v_a, v_b, v_c = int(triangles[i, 0].item()), int(triangles[i, 1].item()), int(
            triangles[i, 2].item())
        v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
        ab, ac = v_b - v_a, v_c - v_a
        cos_ab_ac = (np.sum(ab * ac) / np.clip(np.sqrt(np.sum(ab ** 2)) * np.sqrt(np.sum(ac ** 2)), a_min=1e-9,
                                               a_max=9999999.)).item()
        sin_ab_ac = math.sqrt(min(max(0., 1. - cos_ab_ac ** 2), 1.))
        cur_area = 0.5 * sin_ab_ac * np.sqrt(np.sum(ab ** 2)).item() * np.sqrt(np.sum(ac ** 2)).item()

        # cur_tri_seg = int(triangles_to_seg_idx[i].item())

        cur_sampled_pts = int(cur_area * pts_per_area)
        cur_sampled_pts = 1 if cur_sampled_pts == 0 else cur_sampled_pts
        # if cur_sampled_pts == 0:

        tmp_x, tmp_y = np.random.uniform(0, 1., (cur_sampled_pts,)).tolist(), np.random.uniform(0., 1., (
        cur_sampled_pts,)).tolist()

        for xx, yy in zip(tmp_x, tmp_y):
            sqrt_xx, sqrt_yy = math.sqrt(xx), math.sqrt(yy)
            aa = 1. - sqrt_xx
            bb = sqrt_xx * (1. - yy)
            cc = yy * sqrt_xx
            cur_pos = v_a * aa + v_b * bb + v_c * cc
            sampled_pcts.append(cur_pos)
            # pts_to_seg_idx.append(cur_tri_seg)

    sampled_pcts = np.array(sampled_pcts, dtype=np.float)
    return sampled_pcts



def load_motion_attributes(attribute_fn, ex_none=False):

    def traverse_part_hierarchy(cur_json, tot_motion_attrs):
        cur_dof_name = cur_json["dof_name"]
        cur_part_info = {
            "dof_name": cur_dof_name,
            "motion_type": cur_json["motion_type"],
            "center": cur_json["center"],
            "direction": cur_json["direction"]
        }
        # if ex_none == False or cur_json["motion_type"] != "none":
        tot_motion_attrs.append(cur_part_info) # swhere is seg idx...
        if "children" in cur_json:
            cur_children = cur_json["children"]
            for child in cur_children:
                tot_motion_attrs = traverse_part_hierarchy(child, tot_motion_attrs)
        return tot_motion_attrs

    rf = open(attribute_fn, "r")
    cur_attri_json = json.load(rf)
    motion_attrs = []
    cur_json = cur_attri_json
    motion_attrs = traverse_part_hierarchy(cur_json, motion_attrs)
    return motion_attrs


def load_vertices_triangles(mesh_fn):
    vertices = []
    surfaces = []
    with open(mesh_fn, "r") as rf:
        for line in rf:
            cur_infos = line.split(" ")
            ty = cur_infos[0]
            if ty == "v":
                pos = [float(zz) for zz in cur_infos[1:]]
                vertices.append(pos)
            elif ty == "f":
                vertex_indices = [int(zz) - 1 for zz in cur_infos[1:]]
                surfaces.append(vertex_indices)
        rf.close()

    vertices = np.array(vertices, dtype=np.float)
    surfaces = np.array(surfaces, dtype=np.long)
    return vertices, surfaces


def load_triangles_to_seg_idx(triangles_to_seg_idx_fn, nparts=None):
    seg_idxes = []
    triangles_to_seg = np.load(triangles_to_seg_idx_fn, allow_pickle=True).item()
    triangles_idxes = list(triangles_to_seg.keys())
    minn_tri_idx, maxx_tri_idx = min(triangles_idxes), max(triangles_idxes)
    for i_tri in range(minn_tri_idx, maxx_tri_idx + 1):
        cur_tri_seg = int(triangles_to_seg[i_tri])
        seg_idxes.append(cur_tri_seg)
    # seg_idx; seg_id
    seg_idxes = np.array(seg_idxes, dtype=np.long)

    #### Use n_parts to restrict the number of loaded segmentations/parts ####
    seg_idx_to_triangle_idxes = {}
    for tri_idx in range(seg_idxes.shape[0]):
        cur_tri_seg_idx = int(seg_idxes[tri_idx].item())
        if nparts is None or cur_tri_seg_idx < nparts:
            if cur_tri_seg_idx not in seg_idx_to_triangle_idxes:
                seg_idx_to_triangle_idxes[cur_tri_seg_idx] = [tri_idx]
            else:
                seg_idx_to_triangle_idxes[cur_tri_seg_idx].append(tri_idx)


    return seg_idxes, seg_idx_to_triangle_idxes


def estimate_normals(pos):
    # pos.size = bz x N x 3

    normals = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.numpy())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    nms = np.array(pcd.normals)
    nms = torch.from_numpy(nms).float()
    return nms

    # for i in range(pos.size(0)):
    #     pts = pos[i].detach().cpu().numpy()
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pts)
    #
    #
    #     normals.append(torch.from_numpy(nms).to(pos.device).float().unsqueeze(0))
    # normals = torch.cat(normals, dim=0)
    # return normals

def compute_rotation_matrix_from_axis_angle(axis, angle):

    u, v, w = axis[0].item(), axis[1].item(), axis[2].item()
    costheta = np.cos(angle)
    sintheta = np.sin(angle)

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    m = np.zeros((3, 3), dtype=np.float)


    # m = torch.zeros((n_angles, na, 3, 3), dtype=torch.float32)  # .cuda()
    # print(uu.size(), costheta.size())
    m[0, 0] = uu + (vv + ww) * costheta
    m[1, 0] = uv * (1 - costheta) + w * sintheta
    m[2, 0] = uw * (1 - costheta) - v * sintheta

    m[0, 1] = uv * (1 - costheta) - w * sintheta
    m[1, 1] = vv + (uu + ww) * costheta
    m[2, 1] = vw * (1 - costheta) + u * sintheta

    m[0, 2] = uw * (1 - costheta) + v * sintheta
    m[1, 2] = vw * (1 - costheta) - u * sintheta
    m[2, 2] = ww + (uu + vv) * costheta
    return m


