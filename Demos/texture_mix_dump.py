def get_indexes(faces, correspondences_f_to_smplx):
    my_set = set()
    indexes = []
    for face in faces:
        if (face[0] in correspondences_f_to_smplx):
            if not (face[0] in my_set):
                my_set.add(face[0])
                indexes.append(face[0])
        if (face[1] in correspondences_f_to_smplx):
            if not (face[1] in my_set):
                my_set.add(face[1])
                indexes.append(face[1])
        if (face[2] in correspondences_f_to_smplx):
            if not (face[2] in my_set):
                my_set.add(face[2])
                indexes.append(face[2])
    return indexes

if __name__ == "__main__":
    from PIL import Image

    # Load and read images
    smplx_texture = Image.open("../data/smplx_texture_m_alb.png") #  256,  256, 3
    flame_texture = Image.open("../data/Javi.png")                # 4096, 4096, 3
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
    merged_image.save('merged_image.png')

    # UV MIX
    from utils.file_op import *
    smpl_uv_indexes = read_uv_faces_id_from_obj("../data/smplx-addon.obj") # From face
    smpl_vert_indexes = read_vertex_faces_id_from_obj("../data/smplx-addon.obj") # From face
    correspondences_f_to_smplx = np.load("../data/SMPL-X__FLAME_vertex_ids.npy")

    ver_idx = get_indexes(smpl_vert_indexes, correspondences_f_to_smplx)
    uv_idx = get_indexes(smpl_uv_indexes, correspondences_f_to_smplx)

    # Esto nos dice, en smplx, qué vertice está emparejado con qué UV, para la cabeza de smplx.
    vertex_idxs = np.array(ver_idx)
    uv_idx = np.array(uv_idx)

    #testing
    from utils.texture_match import get_smplx_flame_crossrespondence_face_ids
    flame_2_smplx_uv_ids, smplx_faces, smplx_uv, flame_faces, flame_uv = get_smplx_flame_crossrespondence_face_ids(
        "../data/smplx-addon.obj", "../data/head_template.obj", "../data/SMPL-X__FLAME_vertex_ids.npy", None)

    smplx_uv = read_uv_coordinates_from_obj("../data/smplx-addon.obj")
    smplx_uv[:, 0] = smplx_uv[:, 0] * 0.5  # new body uv coords


    for id in flame_2_smplx_uv_ids.keys():
        f_uv_id = id
        s_uv_id = flame_2_smplx_uv_ids[id]

        flame_idx = flame_faces[f_uv_id]
        smplx_idx = smplx_faces[s_uv_id]

        smplx_uv[smplx_idx, 1] = 1 - flame_uv[flame_idx, 1]
        smplx_uv[smplx_idx, 0] = (flame_uv[flame_idx, 0] * 0.5) + 0.5



    smplx_obj = "../data/smplx-addon.obj"
    vertices = read_vertex_from_obj("../data/IMG_0392_inputs.obj")
    faces = read_vertex_faces_id_from_obj(smplx_obj)
    uvfaces = read_uv_faces_id_from_obj(smplx_obj)
    from utils.texture_match import *
    smplx_texture = cv2.imread('merged_image.png')
    write_obj("TESTING", vertices, faces, texture=smplx_texture, uvcoords=smplx_uv, uvfaces=uvfaces)