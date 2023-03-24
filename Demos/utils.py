# Imports
import torch
import numpy as np
import open3d as o3d
import os

def visualize_meshes(mesh_vertices, mesh_faces, visualize=False, head_idxs=None, head_color=None):
    if visualize is False or (len(mesh_vertices) != len(mesh_faces)):
        return
    complete_meshes = []
    for i in range(len(mesh_vertices)):
        if torch.is_tensor(mesh_vertices[i]) and mesh_vertices[i].requires_grad==True:
            mesh_vertices[i] = mesh_vertices[i].detach().numpy()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices[i])
        mesh.triangles = o3d.utility.Vector3iVector(mesh_faces[i])
        mesh.compute_vertex_normals()

        # Paint Head
        if head_color is not None and head_idxs is not None:
            colors = np.ones_like(mesh_vertices[i]) * [0.3, 0.3, 0.3]
            colors[head_idxs] = head_color
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        complete_meshes.append(mesh)

    o3d.visualization.draw_geometries(complete_meshes)



def head_smoothing(deca_head, smplx_head, head_idx):
    # Weight loading
    abs_path = os.path.abspath('mask_1')
    weights = np.fromfile(abs_path, 'float32')

    # Calculations
    head_weights = torch.tensor(weights[head_idx])
    new_head = torch.zeros_like(deca_head)
    for i in range (5023):
        for j in range (3):
            smpl_coord = smplx_head[i, j] * head_weights[i]
            deca_coord = deca_head[i, j] * (1 - head_weights[i])
            new_head[i, j] = smpl_coord + deca_coord

    return new_head


def save_obj(filename, vertices, faces):
    from smplx_deca_main.smplx.smplx import utils
    utils.write_obj(filename, vertices, faces)
    print("Model Saved")


def getMeshGravityCenter(mesh):
    if not torch.is_tensor(mesh):
        mesh = torch.tensor(mesh)
    summed_coords = torch.sum(mesh, dim=0)
    gravity_center = torch.div(summed_coords, mesh.shape[0])
    return gravity_center


def applyManualTransform(vertices):
    # Apply manual transformations if desired
    # Add offset to y coordinate
    #vertices[:, 1] += 0.15
    # Add offset to z coordinate
    #vertices[head_idxs, 2] += 0.045
    # Add scaling
    #vertices[head_idxs, :] *= 0.5
    #vertices[:] *= 1.2
    return vertices


def create_local_rotation(x_degrees = 0, y_degrees = 0, z_degrees = 0):
    global_orient = torch.zeros([1, 3], dtype=torch.float32)
    global_orient[0, 0] = x_degrees
    global_orient[0, 1] = y_degrees
    global_orient[0, 2] = z_degrees
    degrees_to_rads = 3.14159/180
    global_orient *= degrees_to_rads
    return global_orient


def create_global_translation(x = 0, y = 0, z = 0):
    global_orient = torch.zeros([1, 3], dtype=torch.float32)
    global_orient[0, 0] = x
    global_orient[0, 1] = y
    global_orient[0, 2] = z
    return global_orient