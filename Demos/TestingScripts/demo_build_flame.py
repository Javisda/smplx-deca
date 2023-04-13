import os, sys
import os.path as osp
import argparse
import torch
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from smplx_deca_main.deca.decalib.deca import DECA
from smplx_deca_main.deca.decalib.datasets import datasets
from smplx_deca_main.deca.decalib.utils.config import cfg as deca_cfg

import smplx_deca_main.smplx as smplx
from smplx_deca_main.smplx.smplx import body_models as smplx

def main(args):

    # --------------------------- INIT PARAMS ---------------------------
    show_meshes = False
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


    # --------------------------- RUN MODELS ---------------------------
    # Create expression
    expression = torch.zeros(1, 10)
    expression[0, 0] = -2

    # ------------ FLAME ------------
    flame_model = smplx.create(model_folder, model_type='flame',
                         gender=gender,
                         ext=ext, expression=expression)

    global_orient = apply_global_orientation(0.0, 90.0, 0.0)
    flame_betas = torch.zeros([1, 10], dtype=torch.float32)
    flame_expression = torch.zeros([1, 10], dtype=torch.float32)

    flame_output = flame_model(betas=flame_betas, expression=flame_expression,
                   return_verts=True,
                   global_orient=global_orient)
    flame_vertices = flame_output.vertices.detach().cpu().numpy().squeeze()

    save_obj('TestSamples/flame.obj', flame_vertices, flame_model.faces)

def show_mesh(vertices, model, head_idxs, head_color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    mesh.compute_vertex_normals()

    colors = np.ones_like(vertices) * [0.3, 0.3, 0.3]
    colors[head_idxs] = head_color

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh])


def applyTransform(vertices, head_idxs):
    # Apply manual transformations if desired
    # Add offset to y coordinate
    #vertices[head_idxs, 1] += 0.295
    # Add offset to z coordinate
    #vertices[head_idxs, 2] += 0.045
    # Add scaling
    #vertices[head_idxs, :] *= 0.5
    vertices[:] *= 0.95

def save_obj(filename, vertices, faces):
    vertices = vertices
    faces = faces
    from smplx_deca_main.smplx.smplx import utils
    utils.write_obj(filename, vertices, faces)
    print("Model Saved")

def apply_global_orientation(x_degrees=0, y_degrees=0, z_degrees=0):
    global_orient = torch.zeros([1, 3], dtype=torch.float32)
    global_orient[0, 0] = x_degrees
    global_orient[0, 1] = y_degrees
    global_orient[0, 2] = z_degrees
    degrees_to_rads = 3.14159 / 180
    global_orient *= degrees_to_rads
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