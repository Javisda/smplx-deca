# Imports
import torch
import numpy as np
import open3d as o3d
import os
import cv2
from PIL import Image

def visualize_meshes(mesh_vertices, mesh_faces, visualize=False, head_idxs=None, head_color=None):
    if visualize is False or (len(mesh_vertices) != len(mesh_faces)):
        return
    complete_meshes = []
    for i in range(len(mesh_vertices)):
        if torch.is_tensor(mesh_vertices[i]):
            if mesh_vertices[i].shape[0] == 1:
                mesh_vertices[i] = mesh_vertices[i].squeeze(dim=0)
            if mesh_vertices[i].requires_grad==True:
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
    abs_path = os.path.abspath('Neck_masks/smoothing_mask_1')
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
def optimize_head_alignment(mesh1, mesh2, step_size=0.00000001, max_iters=1000, max_iters_without_improvement=20):

    # Check input integrity
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
        #print("Distance loss: " + str(loss))

        # Update the best loss and mesh checkpoint if needed
        if loss < best_loss:
            best_loss = loss
            best_alignment_checkpoint = mesh1.clone()
            current_iter_without_improvement = 0
        else:
            current_iter_without_improvement += 1

        # Check if optimization isn't improving for a number of steps
        if (current_iter_without_improvement == max_iters_without_improvement) or (step == (max_iters - 1)):
            break

    coords_to_optimize.requires_grad = False
    # Return optimized mesh1
    return best_alignment_checkpoint, step

def learn_body_from_head(head, smpl_model, head_idxs):

    print("-> Body inference")

    # 1º Get head shape to learn body from
    head_shape = head

    # 2º Hyper-parameters
    lr = 0.2
    current_iters = 0
    consecutive_iters_checkpoint = 20
    shape = smpl_model.betas
    shape.requires_grad = True
    finished = False
    debug_anomalies = False
    optimizer = torch.optim.Adam([shape], lr=lr)
    checkpoint_error = None
    best_error = 1e20
    max_iters = 1000
    best_betas = torch.zeros(10, dtype=torch.float32)

    # Tensorboard Initialization
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'tensorboard/tensorboard_test')
    step = 0

    # Optimization Loop
    for epoch in range(0, max_iters):
        save = False
        current_iters += 1
        optimizer.zero_grad()

        # 3º Body generation with best found betas at the moment
        smpl_training_output = smpl_model(betas=shape, return_verts=True)

        # 4º Get SMPLX generated head
        smpl_training_head = smpl_training_output['v_shaped'].squeeze(dim=0)

        # 5º Find head to head euclidean distance without sqrt (function loss)
        loss = (head_shape - smpl_training_head[head_idxs]).pow(2).sum()

        # 6º Calculate gradients, make a step in optimizer and early stoping in case of already found good betas
        # print(f"Loss: {loss}")
        if loss < 0.005:
            finished = True

        if best_error > loss.item():
            save = True

        best_error = min(best_error, loss.item())
        if finished:
            break

        if epoch % consecutive_iters_checkpoint == 0:
            if checkpoint_error is not None:
                if best_error > checkpoint_error * 0.99:
                    break
            checkpoint_error = best_error
        if debug_anomalies:
            with torch.autograd.detect_anomaly():
                loss.backward(retain_graph=True)  # calculate derivatives
        else:
            loss.backward(retain_graph=True)  # calculate derivatives
        optimizer.step()

        if save:
            best_betas = shape

        # Tensorboard writing
        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1

    print("-> Best Betas: ", [round(beta.item(), 5) for beta in best_betas[0]])
    print("-> Mejor loss: {}", round(best_error, 5))
    print("-> End body inference")
    return best_betas

def input_identities(names_of_identities):
    absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    identity_path = os.path.join(absolute_path, 'smplx-deca', 'decaTestSamples', 'examples')
    paths = []
    for identity in names_of_identities:
        path = os.path.join(identity_path, identity)
        paths.append(path)

    return paths

def input_expressions(names_of_expressions):
    absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    identity_path = os.path.join(absolute_path, 'smplx-deca', 'decaTestSamples', 'exp')
    paths = []
    for expression in names_of_expressions:
        path = os.path.join(identity_path, expression)
        paths.append(path)

    return paths

def transfer_texture_information(id_opdict, auxiliar_opdict):
    for texture_name in ['uv_texture_gt', 'rendered_images', 'alpha_images',
                         'normal_images', 'albedo', 'uv_texture', 'normals',
                         'uv_detail_normals', 'displacement_map']:
        id_opdict[texture_name] = auxiliar_opdict[texture_name]

    return id_opdict

def pose_model():
    body_pose = torch.zeros([1, 63], dtype=torch.float32)
    # 1º joint left leg
    body_pose[0,  :3] = create_local_rotation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)
    # 1º joint rigth leg
    body_pose[0, 3:6] = create_local_rotation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)
    # pelvis
    body_pose[0, 6:9] = create_local_rotation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)

    return body_pose


def generate_mixed_uvs(correspondences_path):
    from uv_mixing_utils import get_smplx_flame_crossrespondence_face_ids, read_uv_coordinates_from_obj

    flame_2_smplx_uv_ids, smplx_faces, smplx_uv, flame_faces, flame_uv = get_smplx_flame_crossrespondence_face_ids(
        "UV_mixing_resources/smplx-addon.obj", "UV_mixing_resources/head_template.obj", correspondences_path, None)

    smplx_uv = read_uv_coordinates_from_obj("UV_mixing_resources/smplx-addon.obj")
    smplx_uv[:, 0] = smplx_uv[:, 0] * 0.5  # new body uv coords

    for id in flame_2_smplx_uv_ids.keys():
        f_uv_id = id
        s_uv_id = flame_2_smplx_uv_ids[id]

        flame_idx = flame_faces[f_uv_id]
        smplx_idx = smplx_faces[s_uv_id]

        smplx_uv[smplx_idx, 1] = 1 - flame_uv[flame_idx, 1]
        smplx_uv[smplx_idx, 0] = (flame_uv[flame_idx, 0] * 0.5) + 0.5

    return smplx_uv


def generate_mixed_textures(flame_tex_path):

    # Read RGB to Image format
    flame_texture = Image.open(flame_tex_path)
    smplx_texture = Image.open("UV_mixing_resources/smplx_texture_m_alb.png")

    # Select same resolution for both images
    width, height = 4096, 4096

    # Resize to the same resolution so they can be combines
    smplx_texture = smplx_texture.resize((width, height))
    flame_texture = flame_texture.resize((width, height))

    # Create a new image with the size of both images combined
    merged_image = Image.new("RGB", (width * 2, height))
    merged_image.paste(smplx_texture, (0, 0))
    merged_image.paste(flame_texture, (width, 0))

    # Save the merged image
    # merged_image.save('merged_smplx.png')

    return merged_image


def neck_smoothing_for_textures(smplx_body):

    #  Index mapping
    #  3218 -> 3936  #  3219 -> 3937  #  3236 -> 3239  #  3237 -> 3238  #  3329 -> 3939  #  3330 -> 3938
    #  4427 -> 4429  #  4428 -> 3376  #  4436 -> 4434  #  4437 -> 3378  #  4438 -> 3379  #  5350 -> 3831
    #  5450 -> 5656  #  5453 -> 3941  #  5454 -> 3942  #  5533 -> 5621  #  5981 -> 6684  #  5982 -> 6685
    #  5999 -> 6002  #  6000 -> 6001  #  6092 -> 6687  #  6093 -> 6686  #  7163 -> 7165  #  7164 -> 6137
    #  7172 -> 7170  #  7173 -> 6139  #  7174 -> 6140  #  8184 -> 8350  #  8187 -> 6689  #  8188 -> 6690

    outer_idxs = [3218, 3219, 3236, 3237, 3329, 3330, 4427, 4428, 4436, 4437,
                  4438, 5350, 5450, 5453, 5454, 5533, 5981, 5982, 5999, 6000,
                  6092, 6093, 7163, 7164, 7172, 7173, 7174, 8184, 8187, 8188]
    inner_idxs = [3936, 3937, 3239, 3238, 3939, 3938, 4429, 3376, 4434, 3378,
                  3379, 3831, 5656, 3941, 3942, 5621, 6684, 6685, 6002, 6001,
                  6687, 6686, 7165, 6137, 7170, 6139, 6140, 8350, 6689, 6690]
    for o_to_i in range(len(outer_idxs)):
        # Finds the mid-point between 2 vertices (from the mapping)
        new_mixed_coords = smplx_body[outer_idxs[o_to_i]] * 0.5 + smplx_body[inner_idxs[o_to_i]] * 0.5

        # Apply new position to both vertices
        smplx_body[outer_idxs[o_to_i]] = new_mixed_coords
        smplx_body[inner_idxs[o_to_i]] = new_mixed_coords

    return smplx_body

def head_rescale(head_to_reshape, smpl_head, faces):
    from torch.nn import Parameter

    s = torch.tensor([1], requires_grad=True, dtype=torch.float)
    model = [Parameter(s)]
    lr = 0.002
    optimizer = torch.optim.Adam(model, lr=lr)

    head_to_reshape = head_to_reshape.detach()
    smpl_head = smpl_head.detach()

    #visualize_meshes([head_to_reshape, smpl_head], [faces, faces], visualize=True)
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = (head_to_reshape * model[0] - smpl_head).pow(2).sum()
        loss.backward(retain_graph=True)
        optimizer.step()
    #visualize_meshes([head_to_reshape * model[0], smpl_head], [faces, faces], visualize=True)

    # return the scaling factor
    return model[0]