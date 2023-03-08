import os, sys
import os.path as osp
import argparse
import torch
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from smplx_deca_main.deca.decalib.deca import DECA
from smplx_deca_main.deca.decalib.datasets import datasets
from smplx_deca_main.deca.decalib.utils.config import cfg as deca_cfg

import smplx_deca_main.smplx as smplx
from smplx_deca_main.smplx.smplx import body_models as smplx

def main(args):

    # --------------------------- MODELS INIT PARAMS ---------------------------
    show_meshes = True
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
    global_orient = apply_global_orientation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)
    # Body pose
    body_pose = torch.zeros([1, 63], dtype=torch.float32)
    # 1º joint left leg
    # body_pose[0, :3] = apply_global_orientation(x_degrees=45.0, y_degrees=45.0, z_degrees=45.0)
    # 1º joint rigth leg
    # body_pose[0, 3:6] = apply_global_orientation(x_degrees=45.0, y_degrees=45.0, z_degrees=45.0)
    # pelvis
    body_pose[0, 6:9] = apply_global_orientation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)

    # Build Body Shape and Face Expression Coefficients
    smpl_betas = torch.ones([1, 10], dtype=torch.float32)
    smpl_expression = torch.zeros([1, 10], dtype=torch.float32)

    smpl_model = smplx.create(model_folder, model_type='smplx',
                         gender=gender,
                         ext=ext)

    smpl_body_template = smpl_model.v_template

    # ------------ CREATE DECA MODEL ------------
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)
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


    # ------------ CALCULATE HEAD VERTICES BASE ON OFFSETS WITH MULTIPLE MODELS ------------
    import pickle
    with open("D:\-DYDDV - URJC\SEDDI\smplx-deca-copia\smplx-deca\smplx_deca_main\deca\data\generic_model.pkl", "rb") as f:
        generic_deca = pickle.load(f, encoding='latin1')

    deca_neutral_vertices = generic_deca['v_template']

    # OFFSETS AMONG NEUTRAL DECA AND NEUTRAL SMPLX HEADS
    # Smpl is posed differently as deca, so first we align them.
    smpl_neutral_head_vertices = smpl_body_template[head_idxs].numpy().copy()


        # Find head gravity centers
    """
    sumed_coords = torch.sum(smpl_body_template[head_idxs], dim=0)
    center_of_gravity_smplx = torch.div(sumed_coords, smpl_body_template[head_idxs].shape[0])
    sumed_coords = torch.sum(torch.tensor(deca_neutral_vertices), dim=0)
    center_of_gravity_deca_neutral = torch.div(sumed_coords, smpl_body_template[head_idxs].shape[0])
    """
    # TESTING
    center_of_gravity_smplx = getGravityCenter(mesh=smpl_body_template[head_idxs])
    center_of_gravity_deca_neutral = getGravityCenter(mesh=deca_neutral_vertices)


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


    # LOOP DE ENTRENAMIENTO
    # 1º Coger cabeza generada en el paso anterior (head_vertices_no_expression)
    deca_head_copy = head_vertices_no_expression
    # 2º Crear modelo que va a registrar las 10 betas e iniciarlas a un valor aleatorio
    b0 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b1 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b2 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b3 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b4 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b5 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b6 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b7 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b8 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    b9 = torch.randn(1, requires_grad=True, dtype=torch.float32)
    from torch import nn
    from torch.nn import Parameter
    model = [Parameter(b0), Parameter(b1),Parameter(b2),Parameter(b3),Parameter(b4),Parameter(b5),Parameter(b6),Parameter(b7),Parameter(b8),Parameter(b9)]
    lr = 0.1
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model, lr=lr)
    previous_loss = 9999999
    best_betas = torch.zeros(10, dtype=torch.float32)
    # --- Loop pasos 3º-8º ---
    for epoch in range(1000):
        optimizer.zero_grad()
        # 3º Con estas betas generar un cuerpo con smplx
        smpl_training_betas = torch.cat((model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7], model[8], model[9]), 0).unsqueeze(0)
        smpl_training_output = smpl_model(betas=smpl_training_betas, return_verts=True)
        # 4º Quedarnos con la cabeza generada con smplx
        smpl_training_head = smpl_training_output['v_shaped'].squeeze(dim=0)
        # 5º Alinear cabezas de los pasos 1 y 4
        center_of_gravity_training_smpl = getGravityCenter(mesh=smpl_training_head[head_idxs])
        center_of_gravity_training_deca = getGravityCenter(mesh=deca_head_copy)
        head_training_offsets = center_of_gravity_training_smpl - center_of_gravity_training_deca
        smpl_training_aligned = torch.sub(torch.tensor(smpl_training_head[head_idxs]), head_training_offsets)
        # 6º Hallar las distancias entre ambas cabezas (function loss)
        distances = torch.sum(torch.sub(deca_head_copy, smpl_training_aligned), dim=1)
        total_distance = distances.sum().unsqueeze(0)
        # 7º Hacer un step en la función mediante el loss del paso 6 y usando el optimizador Adam
        loss = criterion(total_distance.float(), torch.zeros((1), dtype=torch.float32))
        #loss = criterion(deca_head_copy, smpl_training_aligned)
        loss.backward()
        optimizer.step()
        if previous_loss > loss:
            previous_loss = loss
            best_betas = smpl_training_betas
            print("MEJORA")
        # 8º Con el paso 7 habremos conseguido nuevos valores de beta, con los cuales repetiremos los pasos anteriores
        print("Loss: " + str(loss))
        print("Updated Betas: ", [beta.item() for beta in model])
    print("Mejor loss: {}", previous_loss)
    # 9º A partir de aquí deberíamos tener unas betas que generan un cuerpo acorde a la cabeza de DECA.
    # Por lo que con estas betas generamos el cuerpo real, pasando al código de abajo ya de forma normal.




    # --------------------- BODY MODEL GENERATION ---------------------
    # GENERATE MODEL WITHOUT EXPRESSION
        # First do a forward pass through the model to get body shape
    smpl_output = smpl_model(betas=best_betas, return_verts=True)
        # Second replace heads. This order is important because head from smpl_output is already modified caused by betas
    body_only_betas = smpl_output['v_shaped'].squeeze(dim=0)
    body_only_betas[head_idxs] = head_vertices_no_expression.float()
    smpl_model.v_template = body_only_betas
        # Third and finally do another forward pass to get final model rotated, posed and translated. Reseting betas to zero is key.
    smpl_betas_zeros = torch.zeros([1, 10], dtype=torch.float32)
    smpl_output = smpl_model(betas=smpl_betas_zeros, expression=smpl_expression,
                   global_orient=global_orient,
                   body_pose=body_pose,
                   transl=global_position)

    smpl_vertices_no_expression = smpl_output.vertices.detach().cpu().numpy().squeeze()
    smpl_joints_body = smpl_output.joints.detach().cpu().numpy().squeeze()

    """
    # Old implementation. Head was influenced by betas and was more adjustable to body proportions.
        smpl_model.v_template[head_idxs] = head_vertices_no_expression.float()
    smpl_output = smpl_model(betas=smpl_betas, expression=smpl_expression,
                   return_verts=True,
                   global_orient=global_orient,
                   body_pose=body_pose,
                   transl=global_position)
    smpl_vertices_no_expression = smpl_output.vertices.detach().cpu().numpy().squeeze()
    smpl_joints_body = smpl_output.joints.detach().cpu().numpy().squeeze()
    """




    # GENERATE MODEL WITH EXPRESSION
        # First isn't necessary anymore as it would shape again the body and accumulate
        # Second replace heads. This order is important because head from smpl_output is already modified caused by betas
    body_only_betas[head_idxs] = head_vertices_expression.float()
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



def getGravityCenter(mesh):
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
    #vertices[:, 1] += 0.295
    # Add offset to z coordinate
    #vertices[head_idxs, 2] += 0.045
    # Add scaling
    #vertices[head_idxs, :] *= 0.5
    #vertices[:] *= 1.1
    return vertices

def save_obj(filename, vertices, faces):
    vertices = vertices
    faces = faces
    from smplx_deca_main.smplx.smplx import utils
    utils.write_obj(filename, vertices, faces)
    print("Model Saved")

def apply_global_orientation(x_degrees = 0, y_degrees=0, z_degrees=0):
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
