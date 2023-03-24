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


def get_mesh_root(mesh):
    if not torch.is_tensor(mesh):
        mesh = torch.from_numpy(mesh)

    # Compute the mean of the vertices in each dimension
    root_x = torch.mean(mesh[:, 0])
    root_y = torch.mean(mesh[:, 1])
    root_z = torch.mean(mesh[:, 2])

    # Create the tensor representing the root
    root = torch.tensor([root_x, root_y, root_z])

    return root
def optimize_head_alignment2(mesh1, mesh2, step_size=0.00000001, max_iters=1000, max_iters_without_improvement=20):

    if not torch.is_tensor(mesh1):
        mesh1 = torch.from_numpy(mesh1)
    if not torch.is_tensor(mesh2):
        mesh2 = torch.from_numpy(mesh2)


    # Get initial roots
    root1 = get_mesh_root(mesh1)
    root2 = get_mesh_root(mesh2)

    # Create params to optimize
    coords_to_optimize = torch.tensor([root1[0], root1[1], root1[2]], requires_grad=True)

    # Create optimizer
    optimizer = torch.optim.Adam([coords_to_optimize], lr=step_size)

    # Optimization loop
    best_loss = float("inf")
    current_iter_without_improvement = 0
    best_alignment_checkpoint = mesh1.clone()
    for step in range(max_iters):
        # Update mesh1 with new root
        mesh1[:, 0] += coords_to_optimize[0] - root1[0]
        mesh1[:, 1] += coords_to_optimize[1] - root1[1]
        mesh1[:, 2] += coords_to_optimize[2] - root1[2]

        # Calculate loss
        loss = (mesh1 - mesh2).pow(2).sum()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Distance loss: " + str(loss))

        # Update the best loss and mesh checkpoint if needed
        if loss < best_loss:
            best_loss = loss
            best_alignment_checkpoint = mesh1.clone()
            current_iter_without_improvement = 0
        else:
            current_iter_without_improvement += 1

        # Check if optimization isn't improving for a number of steps
        if current_iter_without_improvement == max_iters_without_improvement:
            break

    coords_to_optimize.requires_grad = False
    # Return optimized mesh1
    return best_alignment_checkpoint