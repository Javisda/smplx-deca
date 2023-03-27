import os, sys
import os.path as osp
import argparse
import torch
import numpy as np
import cv2
from torch.nn import Parameter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from smplx_deca_main.deca.decalib.deca import DECA
from smplx_deca_main.deca.decalib.datasets import datasets
from smplx_deca_main.deca.decalib.utils.config import cfg as deca_cfg
from smplx_deca_main.deca.decalib.utils import util

import smplx_deca_main.smplx as smplx
from smplx_deca_main.smplx.smplx import body_models as smplx

import utils

def main(args):

    # --------------------------- MODELS INIT PARAMS ---------------------------
    show_meshes = True
    use_renderer = True
    learn_body = False
    # ------------ SMPLX ------------
    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    corr_fname = args.corr_fname
    gender = args.gender
    ext = args.ext
    head_color = args.head_color

    # Load smplx-deca head correspondencies
    head_idxs = np.load(corr_fname)

    # ------------ DECA ------------
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # Select identity images
    identity_1 = '/home/javiserrano/Git/smplx-deca/decaTestSamples/examples/IMG_0392_inputs.jpg'
    identity_2 = '/home/javiserrano/Git/smplx-deca/decaTestSamples/examples/5.jpg'
    identities = []
    identities.append(identity_1)
    identities.append(identity_2)

    # Select expression images
    expression_1 = '/home/javiserrano/Git/smplx-deca/decaTestSamples/exp/7.jpg'
    expression_2 = '/home/javiserrano/Git/smplx-deca/decaTestSamples/exp/0.jpg'
    expressions = []
    expressions.append(expression_1)
    expressions.append(expression_2)

    # Select bodies
    smpl_betas_1 = torch.tensor([0.3776465356349945,
                               -1.1383668184280396,
                               3.765796422958374,
                               -3.6816511154174805,
                               -1.0226212739944458,
                               -3.976848602294922,
                               -4.116629123687744,
                               1.602636456489563,
                               -1.5878002643585205,
                               -1.6307952404022217], dtype=torch.float32).unsqueeze(0)
    smpl_betas_2 = torch.ones([1, 10], dtype=torch.float32) * (-1)
    body_shapes = []
    body_shapes.append(smpl_betas_1)
    body_shapes.append(smpl_betas_2)

    # Select genders
    gender_1 = 'male'
    gender_2 = 'female'
    genders = []
    genders.append(gender_1)
    genders.append(gender_2)

    for j in range(len(identities)):

        identity_path = identities[j]
        expression_path = expressions[j]
        body_shape_parameters = body_shapes[j]
        args.gender = genders[j]

        # load test images
        testdata = datasets.TestData(identity_path, iscrop=args.iscrop, face_detector=args.detector)
        expdata = datasets.TestData(expression_path, iscrop=args.iscrop, face_detector=args.detector)


        # ------------ CREATE SMPLX MODEL ------------w
        # Body Translation
        global_position = utils.create_global_translation(x=0.0, y=0.0, z=0.0)
        # Body orientation
        global_orient = utils.create_local_rotation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)
        # Body pose
        body_pose = torch.zeros([1, 63], dtype=torch.float32)
        # 1º joint left leg
        # body_pose[0, :3] = create_local_rotation(x_degrees=45.0, y_degrees=45.0, z_degrees=45.0)
        # 1º joint rigth leg
        # body_pose[0, 3:6] = create_local_rotation(x_degrees=45.0, y_degrees=45.0, z_degrees=45.0)
        # pelvis
        body_pose[0, 6:9] = utils.create_local_rotation(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0)

        # Build smplx init model
        smpl_expression = torch.zeros([1, 10], dtype=torch.float32)
        smpl_model = smplx.create(model_folder, model_type='smplx',
                             gender=gender,
                             ext=ext)
        smpl_output = smpl_model(betas=body_shape_parameters, return_verts=True)

        smpl_body_generated = smpl_output['v_shaped'].squeeze(dim=0)

        # ------------ CREATE DECA MODEL ------------
        deca_cfg.model.extract_tex = args.extractTex
        deca_cfg.model.use_tex = args.useTex
        deca_cfg.rasterizer_type = args.rasterizer_type
        deca = DECA(config=deca_cfg, device=device, use_renderer=use_renderer)
        # identity reference
        id = 0
        name = testdata[id]['imagename']
        name_exp = expdata[id]['imagename']
        images = testdata[id]['image'].to(device)[None, ...]

        # ------------ RUN DECA TO GENERATE HEAD MODELS ------------
        # Get dict to generate no expression head model
        with torch.no_grad():
            id_codedict = deca.encode(images)
        # Clear expresion and pose parameters from DECA generated head as it will help fitting the model better.
        id_codedict['exp'] = torch.zeros((1, 50))
        id_codedict['pose'] = torch.zeros((1, 6))
        id_codedict['cam'] = torch.tensor((9.5878, 0.0057931, 0.024715), dtype=torch.float32)
        id_opdict, id_visdict = deca.decode(id_codedict)
        id_visdict = {x: id_visdict[x] for x in ['inputs', 'shape_detail_images']}

        # Get dict to generate expression head model
        # -- expression transfer
        # exp code from image
        exp_images = expdata[id]['image'].to(device)[None, ...]
        with torch.no_grad():
            exp_codedict = deca.encode(exp_images)

        # transfer exp code
        id_codedict['pose'][:, 3:] = exp_codedict['pose'][:, 3:]
        id_codedict['exp'] = exp_codedict['exp']
        transfer_opdict, transfer_visdict = deca.decode(id_codedict)
        id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
        os.makedirs(os.path.join(savefolder, name, 'images'), exist_ok=True)
        cv2.imwrite(os.path.join(savefolder, name, 'images/animation.jpg'), deca.visualize(id_visdict))

        transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name, 'deca_head/reconstruction'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, name, 'deca_head/animation'), exist_ok=True)


        # ------------ CALCULATE HEAD VERTICES BASED ON OFFSETS WITH MULTIPLE HEAD MODELS ------------
        path_to_neutral_deca = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                                            'smplx-deca', 'smplx_deca_main', 'deca', 'data', 'generic_model.pkl')
        import pickle
        with open(path_to_neutral_deca, "rb") as f:
            generic_deca = pickle.load(f, encoding='latin1')

        deca_neutral_vertices = generic_deca['v_template']
        deca_neutral_vertices = deca_neutral_vertices.astype(np.float32)


        # 1º Calculate offsets between pairs of heads
        generated_deca_head_no_expression = id_opdict['verts'].detach().clone()
        generated_deca_head_expression = transfer_opdict['verts'].detach().clone()
        deca_neutral_head = deca_neutral_vertices
        smplx_neutral_head = smpl_model.v_template[head_idxs]
        generated_smplx_head = smpl_body_generated[head_idxs]
        heads_to_align = [generated_deca_head_expression, generated_deca_head_no_expression, deca_neutral_head, smplx_neutral_head, generated_smplx_head]
        total_offset_no_expression = torch.zeros_like(smplx_neutral_head, dtype=torch.float32)
        total_offset_expression = torch.zeros_like(smplx_neutral_head, dtype=torch.float32)
        for i in range(len(heads_to_align) - 1):
            head_1 = heads_to_align[i]
            head_2 = heads_to_align[i + 1]

            # Find head gravity centers
            gravity_center_1 = utils.get_mesh_root(mesh=head_1)
            gravity_center_2 = utils.get_mesh_root(mesh=head_2)

            # Get root to root offsets (having gravity centers as reference)
            root_to_root_offsets = torch.sub(gravity_center_1, gravity_center_2)

            # Having the offsets, make an initial guess of alignment
            if not torch.is_tensor(head_1):
                head_1 = torch.from_numpy(head_1)
            head_1_aligned = head_1 - root_to_root_offsets

            # Visualize first alignment based on gravity center
            utils.visualize_meshes([head_2, head_1_aligned], [generic_deca['f'], generic_deca['f']],
                                   visualize=False)

            # Better alignment raining loop
            best_head_alignment = utils.optimize_head_alignment(head_1_aligned, head_2)

            # Visualize second alignment after optimization
            utils.visualize_meshes([head_2, head_1_aligned], [generic_deca['f'], generic_deca['f']],
                                   visualize=False)

            # After optimization, calculate shape offsets among both faces
            shape_offset_deca_and_smplx_neutrals = torch.sub(best_head_alignment,
                                                             torch.tensor(head_2))

            # Accumulate offset
            if i == 0:
                total_offset_expression += shape_offset_deca_and_smplx_neutrals.squeeze(0)
            else:
                total_offset_expression += shape_offset_deca_and_smplx_neutrals.squeeze(0)
                total_offset_no_expression += shape_offset_deca_and_smplx_neutrals.squeeze(0)


        # NORMAL HEAD (NO EXPRESSION)
            # 2º Select smplx shape
        normal_body_vertices = smpl_body_generated
            # 3º Select smplx head shape
        smplx_head = normal_body_vertices[head_idxs, :]
            # 4º Add offsets to smplx head
        smplx_head_with_offsets = smplx_head + total_offset_no_expression
            # 5º Replace vertices
        head_vertices_no_expression = smplx_head_with_offsets


        # EXPRESSION BODY
            # 2º Select smplx shape
        expression_body_vertices = smpl_body_generated
            # 3º Select smplx head shape
        smplx_head = expression_body_vertices[head_idxs, :]
            # 4º Add offsets to smplx head
        smplx_head_with_offsets = smplx_head + total_offset_expression
            # 5º Replace vertices
        head_vertices_expression = smplx_head_with_offsets


        # LEARN BODY FROM HEAD SHAPE
        if learn_body:
            best_betas = utils.learn_body_from_head(head_vertices_no_expression, smpl_model, head_idxs)
            # Do a forward pass through the model to get body shape
            smpl_output = smpl_model(betas=best_betas, return_verts=True)


        # --------------------- BODY MODEL GENERATION ---------------------
        # GENERATE MODEL WITHOUT EXPRESSION
            # 1º Replace smplx head for deca head and smooth neck.
        smplx_body = smpl_output['v_shaped'].squeeze(dim=0)
        smplx_body[head_idxs] = utils.head_smoothing(head_vertices_no_expression.float(), smplx_body[head_idxs], head_idx=head_idxs) # Comment this to get the smplx body with the head that best matches deca head
        smpl_model.v_template = smplx_body
            # 2º Do another forward pass to get final model rotated, posed and translated. Reseting betas to zero is key.
        smpl_betas_zeros = torch.zeros([1, 10], dtype=torch.float32)
        smpl_output = smpl_model(betas=smpl_betas_zeros, expression=smpl_expression,
                       global_orient=global_orient,
                       body_pose=body_pose,
                       transl=global_position)

        smpl_vertices_no_expression = smpl_output.vertices.detach().cpu().numpy().squeeze()
        smpl_joints_body = smpl_output.joints.detach().cpu().numpy().squeeze()


        # GENERATE MODEL WITH EXPRESSION
             # 1º Replace smplx head for deca head and smooth neck.
        smplx_body[head_idxs] = utils.head_smoothing(head_vertices_expression.float(), smplx_body[head_idxs], head_idx=head_idxs)
        smpl_model.v_template = smplx_body
            # 2º Do another forward pass to get final model rotated, posed and translated. Reseting betas to zero is key.
        smpl_output = smpl_model(betas=smpl_betas_zeros, expression=smpl_expression,
                       global_orient=global_orient,
                       body_pose=body_pose,
                       transl=global_position)
        smpl_vertices_expression = smpl_output.vertices.detach().cpu().numpy().squeeze()




        # --------------------------- OUTPUT INFO ---------------------------
        print('Vertices shape (SMPLX) =', smpl_vertices_no_expression.shape)
        print('Joints shape (SMPLX) =', smpl_joints_body.shape)


        # --------------------------- SAVE MODELS ---------------------------
        for save_type in ['reconstruction', 'animation']:

            # Save DECA head
            if save_type == 'reconstruction':
                visdict = id_codedict;
                opdict = id_opdict
            else:
                visdict = transfer_visdict;
                opdict = transfer_opdict
            if args.saveDepth:
                depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
                visdict['depth_images'] = depth_image
                cv2.imwrite(os.path.join(savefolder, name, save_type, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
            if args.saveObj:
                deca.save_obj(os.path.join(savefolder, name + '/deca_head', save_type, name + '.obj'), opdict)
            if args.saveMat:
                opdict = util.dict_tensor2npy(opdict)
                from scipy.io import savemat
                savemat(os.path.join(savefolder, name, save_type, name + '.mat'), opdict)
            if args.saveImages:
                for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                    if vis_name not in visdict.keys():
                        continue
                    cv2.imwrite(os.path.join(savefolder, name, save_type, name + '_' + vis_name + '.jpg'), util.tensor2image(visdict[vis_name][0]))
            # -----------------------------------------------------

            # Full body model saving
            if args.saveObj:
                os.makedirs(os.path.join(savefolder, name, 'body_models'), exist_ok=True)
                save_path = os.path.join(savefolder, name + '/body_models/' + save_type +'_exp'+name_exp+'.obj')
                if save_type == 'reconstruction':
                    utils.save_obj(save_path, smpl_vertices_no_expression, smpl_model.faces)
                    utils.visualize_meshes([smpl_vertices_no_expression], [smpl_model.faces], visualize=show_meshes, head_idxs=head_idxs, head_color=head_color)
                else:
                    utils.save_obj(save_path, smpl_vertices_expression, smpl_model.faces)
                    utils.visualize_meshes([smpl_vertices_expression], [smpl_model.faces], visualize=show_meshes, head_idxs=head_idxs, head_color=head_color)
                if os.path.exists(save_path):
                    print(f'Successfully saved {save_path}')
                else:
                    print(f'Error: Failed to save {save_path}')






if __name__ == '__main__':
    # DECA RELATIVE
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp/7.jpg', type=str,
                        help='path to expression')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    # process test images
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode')
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

    # SAVE GENERATED DOCUMENTS RELATIVE
    parser.add_argument('-s', '--savefolder', default='TestSamples', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')


    main(parser.parse_args())