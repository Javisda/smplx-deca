class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

if __name__ == '__main__':
    import pickle

    with open("/home/javiserrano/Git/smplx-deca/smplx_deca_main/deca/data/generic_model.pkl", "rb") as f:
        ss = pickle.load(f, encoding='latin1')
        flame_model = Struct(**ss)

    import open3d as o3d

    # GET FLAME GENERIC HEAD

    vertices = ss['v_template']
    faces = ss['f']

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh])
