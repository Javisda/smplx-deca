import numpy as np
import cv2

def read_vertex_from_obj(fname):
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('v '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:4]]
                res.append(v)

    return np.array(res, dtype=np.float)  # [N,4]


def read_uv_coordinates_from_obj(fname):
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('vt '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:3]]
                res.append(v)
    return np.array(res, dtype=np.float)


def read_vertex_faces_id_from_obj(fname):  # read vertices id in faces: (vv1,vv2,vv3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [int(i) for i in tmp[1:4]]
                res.append(v)
    return np.array(res, dtype=np.int) - 1  # obj index from 1


def read_uv_faces_id_from_obj(fname):  # read texture id in faces: (vt1,vt2,vt3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[1]) for i in tmp[1:4]]
                else:
                    raise (Exception("not a textured obj file"))
                res.append(v)
    return np.array(res, dtype=np.int) - 1  # obj index from 1

def get_smplx_flame_crossrespondence_face_ids(smplx_template_obj,
                                              flame_template_obj,
                                              smplx_flame_vertex_ids,
                                              smplx_face_indexes=None):
    '''
    input:
        smplx_template_obj: smplx template obj /to/path/file.obj
        flame_template_obj: flame template obj /to/path/file.obj
        smplx_flame_vertex_ids: the smplx vertices id crossresponding to flame
    output:
        flame_2_smplx_uv_ids: {flame_uv_id:smplx_uv_id,....}
        s_f_uvs: smplx uv faces
        s_uv: smplx uv coordinates
        f_f_uvs: flame uv faces
        f_uv: flame uv coordinates.
    '''

    # get smplx info from smplx template obj file.
    s_f_ids = read_vertex_faces_id_from_obj(smplx_template_obj)
    s_f_uvs = read_uv_faces_id_from_obj(smplx_template_obj)
    s_uv = read_uv_coordinates_from_obj(smplx_template_obj)

    s_uv[:, 1] = 1 - s_uv[:, 1]  # y--v

    # get flame info from flame template obj file.
    f_verts = read_vertex_from_obj(flame_template_obj)
    f_f_ids = read_vertex_faces_id_from_obj(flame_template_obj)
    f_f_uvs = read_uv_faces_id_from_obj(flame_template_obj)
    f_uv = read_uv_coordinates_from_obj(flame_template_obj)

    f_uv[:, 1] = 1 - f_uv[:, 1]  # y--v

    # smplx to flame vertex ids
    sf_ids = np.load(smplx_flame_vertex_ids)

    if smplx_face_indexes is not None:
        # filtered other index but face vertices index
        face_vertex_ids = np.load(smplx_face_indexes)
        for j in range(f_verts.shape[0]):
            if sf_ids[j] in face_vertex_ids[0]:
                continue
            else:
                sf_ids[j] = -1

    f_2_s_verts_ids = {}
    for ii in range(f_verts.shape[0]):
        f_2_s_verts_ids[ii] = sf_ids[ii]

    # smplx vertex id to face id
    smplx_faces = {}
    for j in range(s_f_ids.shape[0]):
        v1, v2, v3 = s_f_ids[j]
        name = str(v1) + '_' + str(v2) + '_' + str(v3)
        smplx_faces[name] = j

    # get the uv map crossrespondences ids
    flame_2_smplx_uv_ids = {}

    for id, flame_face in enumerate(f_f_ids):
        v1 = f_2_s_verts_ids[flame_face[0]]
        v2 = f_2_s_verts_ids[flame_face[1]]
        v3 = f_2_s_verts_ids[flame_face[2]]
        name = str(v1) + '_' + str(v2) + '_' + str(v3)
        if name in smplx_faces:
            flame_2_smplx_uv_ids[id] = smplx_faces[name]

    return flame_2_smplx_uv_ids, s_f_uvs, s_uv, f_f_uvs, f_uv

def affine_transform(p1, p2, tex1, tex2):
    tex1 = tex1.copy()
    tex2 = tex2.copy()
    # crop tex1 by p1 --> tex2 by p2
    p1 = p1.reshape(-1, 2).astype(np.float32)  # [3,2]
    p2 = p2.reshape(-1, 2).astype(np.float32)  # [3,2]p1 = p1.reshape(-1, 2).astype(np.float32)  # [3,2]
    mat_trans = cv2.getAffineTransform(p1, p2)
    p1 = p1.astype(np.int)
    p2 = p2.astype(np.int)


    dst_h, dst_w, _ = tex2.shape

    #
    tex1_mask = np.zeros(tex1.shape, dtype=np.int8)
    tex2_mask = np.ones(tex2.shape, dtype=np.int8)

    tex1_mask = cv2.drawContours(tex1_mask, [p1], 0, (1, 1, 1), -1)  # 1
    tex2_mask = cv2.drawContours(tex2_mask, [p2], 0, (0, 0, 0), -1)  # 0

    for mask, tex in zip([tex1_mask, tex2_mask], [tex1, tex2]):
        invalid_index = mask == 0
        tex[invalid_index] = 0

    dst = cv2.warpAffine(tex1, mat_trans, (dst_w, dst_h))


    # combine tex2 and dst image
    out_img = dst + tex2

    return out_img