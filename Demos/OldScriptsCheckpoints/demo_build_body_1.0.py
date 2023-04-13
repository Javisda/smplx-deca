import os, sys
import os.path as osp
import argparse
import torch
import numpy as np
import open3d as o3d
import enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from smplx_deca_main.deca.decalib.deca import DECA
from smplx_deca_main.deca.decalib.datasets import datasets
from smplx_deca_main.deca.decalib.utils.config import cfg as deca_cfg

import smplx_deca_main.smplx as smplx
from smplx_deca_main.smplx.smplx import body_models as smplx

def main(args):

    # --------------------------- MODELS INIT PARAMS ---------------------------
    show_meshes = True
    use_renderer = False
    # ------------ SMPLX ------------
    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    corr_fname = args.corr_fname
    gender = args.gender
    ext = args.ext
    head = args.head
    head_color = args.head_color

    # Load smplx-deca head correspondencies
    head_idxs = np.load(corr_fname)

    # ------------ DECA ------------
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)


    # ------------ CREATE SMPLX MODEL ------------w
    # Body Translation
    global_position = apply_global_translation(x=0.0, y=0.0, z=0.0)
    # Body orientation
    global_orient = apply_rotation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)
    # Body pose
    body_pose = torch.zeros([1, 63], dtype=torch.float32)
    # 1º joint left leg
    # body_pose[0, :3] = apply_rotation(x_degrees=45.0, y_degrees=45.0, z_degrees=45.0)
    # 1º joint rigth leg
    # body_pose[0, 3:6] = apply_rotation(x_degrees=45.0, y_degrees=45.0, z_degrees=45.0)
    # pelvis
    body_pose[0, 6:9] = apply_rotation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)

    # Build Body Shape and Face Expression Coefficients
    smpl_betas = torch.zeros([1, 10], dtype=torch.float32)
    smpl_expression = torch.zeros([1, 10], dtype=torch.float32)

    smpl_model = smplx.create(model_folder, model_type='smplx',
                         gender=gender,
                         ext=ext)

    smpl_body_template = smpl_model.v_template

    # ------------ CREATE DECA MODEL ------------
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device, use_renderer=use_renderer)
    # identity reference
    i = 0
    name = testdata[i]['imagename']
    name_exp = expdata[i]['imagename']
    images = testdata[i]['image'].to(device)[None, ...]

    # ------------ RUN DECA TO GENERATE HEAD MODELS ------------
    # Get dict to generate no expression head model
    with torch.no_grad():
        id_codedict = deca.encode(images)
    id_opdict = deca.decode(id_codedict, rendering=False, vis_lmk=False, use_detail=False, return_vis=False)

    # Get dict to generate expression head model
    # -- expression transfer
    # exp code from image
    exp_images = expdata[i]['image'].to(device)[None, ...]
    with torch.no_grad():
        exp_codedict = deca.encode(exp_images)

    # transfer exp code
    id_codedict['pose'][:, 3:] = exp_codedict['pose'][:, 3:]
    id_codedict['exp'] = exp_codedict['exp']
    transfer_opdict = deca.decode(id_codedict, rendering=False, vis_lmk=False, use_detail=False, return_vis=False)


    # ------------ CALCULATE HEAD VERTICES BASED ON OFFSETS WITH MULTIPLE MODELS ------------
    import pickle
    with open("/smplx_deca_main/deca/data/generic_model.pkl", "rb") as f:
        generic_deca = pickle.load(f, encoding='latin1')

    deca_neutral_vertices = generic_deca['v_template']

    # OFFSETS AMONG NEUTRAL DECA AND NEUTRAL SMPLX HEADS
    # Smpl is posed differently as deca, so first we align them.
    smpl_neutral_head_vertices = smpl_body_template[head_idxs].numpy().copy()


        # Find head gravity centers
    center_of_gravity_smplx = getMeshGravityCenter(mesh=smpl_body_template[head_idxs])
    center_of_gravity_deca_neutral = getMeshGravityCenter(mesh=deca_neutral_vertices)


        # Get head to head offsets (having gravity centers as reference)
    smpl_base_to_deca_coords_offsets = center_of_gravity_smplx - center_of_gravity_deca_neutral
        # Having the offsets, translate the smplx head to align deca
    smpl_neutral_aligned = torch.sub(torch.tensor(smpl_neutral_head_vertices), smpl_base_to_deca_coords_offsets)
        # After being aligned, calculate shape offsets among both faces
    shape_offset_deca_and_smplx_neutrals = torch.sub(smpl_neutral_aligned, torch.tensor(deca_neutral_vertices))



    # NORMAL HEAD (NO EXPRESSION)
    normal_body_vertices = smpl_body_template

        # 2º offsets deca generated mesh and deca neutral
    neutral_vertices = generic_deca['v_template']
    normal_deca_offsets = id_opdict['verts'] - neutral_vertices
    normal_deca_offsets = normal_deca_offsets.squeeze(0)

        # 3º Add to full body
    selected_vertices = normal_body_vertices[head_idxs, :]
        # Add offsets
    selected_vertices = selected_vertices + normal_deca_offsets
    selected_vertices = selected_vertices - shape_offset_deca_and_smplx_neutrals # If next offsets are commented, final body won't have lips correction but will have better neck
        # Apply some optional transforms
    selected_vertices = applyTransform(selected_vertices)
        # Replace vertices
    head_vertices_no_expression = selected_vertices


    # EXPRESSION BODY
    expression_body_vertices = smpl_body_template

        # 2º offsets deca generated mesh and deca neutral
    neutral_vertices = generic_deca['v_template']
    exp_deca_offsets = transfer_opdict['verts'] - neutral_vertices
    exp_deca_offsets = exp_deca_offsets.squeeze(0)

        # 3º Add to full body
    selected_vertices = expression_body_vertices[head_idxs, :]
    # Add offsets
    selected_vertices = selected_vertices + exp_deca_offsets
    selected_vertices = selected_vertices - shape_offset_deca_and_smplx_neutrals  # If next offsets are commented, final body won't have lips correction but will have better neck
        # Apply some optional transforms
    selected_vertices = applyTransform(selected_vertices)
        # Replace vertices
    head_vertices_expression = selected_vertices


    # BODY FITTING
    # 1º Get DECA generated head to approximate body
    deca_head_copy = head_vertices_no_expression
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
    body_template_smplx = smpl_model.v_template.clone()

    # Tensorboard Initialization
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'../tensorboard/tensorboard_test')
    step = 0

    # Optimization Loop
    for epoch in range(0, max_iters):
        save = False
        current_iters += 1
        optimizer.zero_grad()

        # 3º Body generation with best found betas at the moment
        #smpl_model.v_template = body_template_smplx # Reestablecer malla del modelo para que no se acumulen las betas
        smpl_training_output = smpl_model(betas=shape, return_verts=True)

        # 4º Get SMPLX generated head
        smpl_training_head = smpl_training_output['v_shaped'].squeeze(dim=0)

        # 5º Find head to head euclidean distance without sqrt (function loss)
        loss = (deca_head_copy - smpl_training_head[head_idxs]).pow(2).sum()

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
                loss.backward()  # calculate derivatives
        else:
            loss.backward()  # calculate derivatives
        optimizer.step()

        if save:
            best_betas = shape


        # Tensorboard writing
        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1

    print("Best Betas: ", [beta.item() for beta in best_betas[0]])
    print("Mejor loss: {}", best_error)




    # --------------------- BODY MODEL GENERATION ---------------------
    # GENERATE MODEL WITHOUT EXPRESSION
        # First do a forward pass through the model to get body shape
    smpl_output = smpl_model(betas=best_betas, return_verts=True)
        # Second replace heads. This order is important because head from smpl_output is already modified caused by betas
    body_only_betas = smpl_output['v_shaped'].squeeze(dim=0)
    body_only_betas[head_idxs] = head_smoothing(head_vertices_no_expression.float(), body_only_betas[head_idxs], head_idx=head_idxs) # Comment this to get the smplx body with the head that best matches deca head
    smpl_model.v_template = body_only_betas
        # Third and finally do another forward pass to get final model rotated, posed and translated. Reseting betas to zero is key.
    smpl_betas_zeros = torch.zeros([1, 10], dtype=torch.float32)
    smpl_output = smpl_model(betas=smpl_betas_zeros, expression=smpl_expression,
                   global_orient=global_orient,
                   body_pose=body_pose,
                   transl=global_position)

    smpl_vertices_no_expression = smpl_output.vertices.detach().cpu().numpy().squeeze()
    smpl_joints_body = smpl_output.joints.detach().cpu().numpy().squeeze()



    # GENERATE MODEL WITH EXPRESSION
        # First isn't necessary anymore as it would shape again the body and accumulate
        # Second replace heads. This order is important because head from smpl_output is already modified caused by betas
    body_only_betas[head_idxs] = head_smoothing(head_vertices_expression.float(), body_only_betas[head_idxs], head_idx=head_idxs)
    smpl_model.v_template = body_only_betas
        # Third and finally do another forward pass to get final model rotated, posed and translated. Reseting betas to zero is key.
    smpl_output = smpl_model(betas=smpl_betas_zeros, expression=smpl_expression,
                   global_orient=global_orient,
                   body_pose=body_pose,
                   transl=global_position)
    smpl_vertices_expression = smpl_output.vertices.detach().cpu().numpy().squeeze()




    # --------------------------- OUTPUT INFO ---------------------------
    print('Vertices shape (SMPLX) =', smpl_vertices_no_expression.shape)
    print('Joints shape (SMPLX) =', smpl_joints_body.shape)


    # --------------------------- SAVE MODELS ---------------------------
    image_name = name
    for save_type in ['reconstruction', 'animation']:
        if args.saveObj:
            save_path = 'TestSamples/' + save_type +'_' +image_name+'_exp'+name_exp+'.obj'
            if save_type == 'reconstruction':
                save_obj(save_path, smpl_vertices_no_expression, smpl_model.faces)
                if show_meshes:
                    show_mesh(smpl_vertices_no_expression, smpl_model, head_idxs, head_color)
            else:
                save_obj(save_path, smpl_vertices_expression, smpl_model.faces)
                if show_meshes:
                    show_mesh(smpl_vertices_expression, smpl_model, head_idxs, head_color)
            if os.path.exists(save_path):
                print(f'Successfully saved {save_path}')
            else:
                print(f'Error: Failed to save {save_path}')


def head_smoothing(deca_head, smplx_head, head_idx):
    # Weight loading
    abs_path = os.path.abspath('../Neck_masks/mask_1')
    weights = np.fromfile(abs_path, 'float32')

    # Calculations
    # First one doesn´t work correctly. Why?
    """
    head_weights = torch.tensor(weights[head_idx]).repeat(1, 3).view(deca_head.shape[0], -1)
    smpl_coords = torch.mul(smplx_head, head_weights)
    deca_coords = torch.mul(deca_head, (torch.ones_like(deca_head) - head_weights))
    new_head = smpl_coords + deca_coords
    """

    head_weights = torch.tensor(weights[head_idx])
    new_head = torch.zeros_like(deca_head)
    for i in range (5023):
        for j in range (3):
            smpl_coord = smplx_head[i, j] * head_weights[i]
            deca_coord = deca_head[i, j] * (1 - head_weights[i])
            new_head[i, j] = smpl_coord + deca_coord

    return new_head

def getMeshGravityCenter(mesh):
    if not torch.is_tensor(mesh):
        mesh = torch.tensor(mesh)
    summed_coords = torch.sum(mesh, dim=0)
    gravity_center = torch.div(summed_coords, mesh.shape[0])
    return gravity_center

def show_mesh(vertices, model, head_idxs, head_color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    mesh.compute_vertex_normals()

    colors = np.ones_like(vertices) * [0.3, 0.3, 0.3]
    colors[head_idxs] = head_color

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh])


def applyTransform(vertices):
    # Apply manual transformations if desired
    # Add offset to y coordinate
    #vertices[:, 1] += 0.15
    # Add offset to z coordinate
    #vertices[head_idxs, 2] += 0.045
    # Add scaling
    #vertices[head_idxs, :] *= 0.5
    #vertices[:] *= 1.2
    return vertices

def save_obj(filename, vertices, faces):
    vertices = vertices
    faces = faces
    from smplx_deca_main.smplx.smplx import utils
    utils.write_obj(filename, vertices, faces)
    print("Model Saved")

def apply_rotation(x_degrees = 0, y_degrees=0, z_degrees=0):
    global_orient = torch.zeros([1, 3], dtype=torch.float32)
    global_orient[0, 0] = x_degrees
    global_orient[0, 1] = y_degrees
    global_orient[0, 2] = z_degrees
    degrees_to_rads = 3.14159/180
    global_orient *= degrees_to_rads
    return global_orient

def apply_global_translation(x = 0, y=0, z=0):
    global_orient = torch.zeros([1, 3], dtype=torch.float32)
    global_orient[0, 0] = x
    global_orient[0, 1] = y
    global_orient[0, 2] = z
    return global_orient


if __name__ == '__main__':
    # DECA RELATIVE
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp/7.jpg', type=str,
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')

    # SMPLX RELATIVE
    parser.add_argument('--model_folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--corr_fname', required=True, type=str,
                        dest='corr_fname',
                        help='Filename with the head correspondences')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--head', default='right',
                        choices=['right', 'left'],
                        type=str, help='Which head to plot')
    parser.add_argument('--head-color', type=float, nargs=3, dest='head_color',
                        default=(0.3, 0.3, 0.6),
                        help='Color for the head vertices')

    main(parser.parse_args())
