# Example optimization with pytorch
# It generates a random shape for a target avatar and tries to fit the
# shape of another avatar to match the target vertices

from smplx_deca_main.smplx.smplx import body_models as smplx  # You need to install smplx first. https://github.com/vchoutas/smplx
import torch
import enum
from typing import Tuple
from pathlib import Path
from collections import namedtuple
import numpy as np


class AvatarType(enum.Enum):
    SMPL = enum.auto()
    SMPLX = enum.auto()


class Avatar:
    def __init__(
        self,
        gender: str,
        avatar_type: AvatarType = AvatarType.SMPL,
        trans: torch.Tensor = None,
        pose: torch.Tensor = None,
        shape: torch.Tensor = None,
    ):
        self._smpl_layer = smplx.create(
            model_type=avatar_type.name,
            gender=gender
        )
        self.gender = gender
        self._avatar_type = avatar_type
        if trans is None:
            trans = torch.zeros(1, 3, dtype=torch.float32)
        if pose is None:
            pose = torch.zeros(1, 72, dtype=torch.float32)
        if shape is None:
            shape = torch.zeros(1, 300, dtype=torch.float32)

        self._trans = trans
        self._pose = pose
        self._shape = shape

    def set_params(
        self, trans: torch.Tensor = None, pose: torch.Tensor = None, shape: torch.Tensor = None
    ):
        if trans is not None:
            self._trans = trans
        if pose is not None:
            self._pose = pose
        if shape is not None:
            self._shape = shape

    def get_params(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self._trans, self._pose, self._shape

    def get_vertices_from_params(
        self,
        a_trans: torch.Tensor,
        a_pose: torch.Tensor,
        a_shape: torch.Tensor,
    ):
        vertices = None
        if self._avatar_type == AvatarType.SMPL:
            """Returns vertices from parameters without modifying the current avatar"""
            output_model: namedtuple = self._smpl_layer(
                transl=a_trans, global_orient=a_pose[:, :3], body_pose=a_pose[:, 3:], betas=a_shape
            )
            vertices = output_model.vertices
        elif self._avatar_type == AvatarType.SMPLX:
            output_model: namedtuple = self._smpl_layer(
                transl=a_trans,
                global_orient=a_pose[:, :3],
                body_pose=a_pose[:, 3:66],
                betas=a_shape,
            )
            vertices = output_model.vertices
        return vertices.squeeze()

    def get_vertices(self) -> torch.Tensor:
        return self.get_vertices_from_params(self._trans, self._pose, self._shape)

    def get_faces(self) -> np.array:
        return self._smpl_layer.faces


def write_obj(
    a_pathToSave: Path,
    a_vertices,
    a_faces,
    a_uvsValues=None,
    a_normalsValues=None,
    a_uvsIDs=None,
    a_normalsIDs=None,
):
    """

    :param a_pathToSave (Path):
    :param a_vertices (np.array([numVertices, 3])):
    :param a_faces (np.array([numFaces, 3])):
    :param a_uvsValues(np.array([numUVs, 2])):
    :param a_normalsValues(np.array([nunNormals, 3])):
    :param a_uvsIDs(np.array([numFaces, 3])):
    :param a_normalsIDs(np.array([numFaces, 3])):
    :return:
    """

    ext = a_pathToSave.suffix
    if ext != ".obj":
        raise ValueError("Error! the extension was {} and should be .obj".format(ext))

    # Faces, uvs ids and normals id are 1-based, not 0-based in obj files
    faceIdMin = np.amin(a_faces)
    faceIdOffset = 1 - faceIdMin  # if the min id was 0 then we only need to add 1
    if faceIdMin != 0 and faceIdMin != 1:
        raise ValueError(
            "Error! wrong format, The face ID doesn't start from zero or 1, starts from {}".format(
                faceIdMin
            )
        )
    if a_uvsIDs is not None:
        uvsIdMin = np.amin(a_faces)
        uvsIdOffset = 1 - uvsIdMin
        if uvsIdMin != 0 and uvsIdMin != 1:
            raise ValueError(
                "Error! wrong format, The uvs ID doesn't start from zero or 1, starts from {}".format(
                    uvsIdMin
                )
            )
    if a_normalsIDs is not None:
        normalsIdMin = np.amin(a_faces)
        normalsIdOffset = 1 - normalsIdMin
        if normalsIdMin != 0 and normalsIdMin != 1:
            raise ValueError(
                "Error! wrong format, The normals ID doesn't start from zero or 1, starts from {}".format(
                    normalsIdMin
                )
            )

    # Checks if folder exists, if not create it
    a_pathToSave.parent.mkdir(parents=True, exist_ok=True)

    with open(a_pathToSave, "w") as fp:

        for v in a_vertices:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))  # x y z
        if a_uvsValues is not None:
            for vt in a_uvsValues:
                fp.write("vt %f %f\n" % (vt[0], vt[1]))  # u v
        if a_normalsValues is not None:
            for vn in a_normalsValues:
                fp.write("vn %f %f %f\n" % (vn[0], vn[1], vn[2]))  # nx ny nz

        current_f = 0
        for f in a_faces:
            if a_uvsIDs is None and a_normalsIDs is None:
                fp.write(
                    "f %d %d %d\n" % (f[0] + faceIdOffset, f[1] + faceIdOffset, f[2] + faceIdOffset)
                )
            if a_uvsIDs is not None and a_normalsIDs is None:
                fp.write(
                    "f %d/%d %d/%d %d/%d\n"
                    % (
                        f[0] + faceIdOffset,
                        a_uvsIDs[current_f, 0] + uvsIdOffset,
                        f[1] + faceIdOffset,
                        a_uvsIDs[current_f, 1] + uvsIdOffset,
                        f[2] + faceIdOffset,
                        a_uvsIDs[current_f, 2] + uvsIdOffset,
                    )
                )
            if a_uvsIDs is None and a_normalsIDs is not None:
                fp.write(
                    "f %d//%d %d//%d %d//%d\n"
                    % (
                        f[0] + faceIdOffset,
                        a_normalsIDs[current_f, 0] + normalsIdOffset,
                        f[1] + faceIdOffset,
                        a_normalsIDs[current_f, 1] + normalsIdOffset,
                        f[2] + faceIdOffset,
                        a_normalsIDs[current_f, 2] + normalsIdOffset,
                    )
                )
            if a_uvsIDs is not None and a_normalsIDs is not None:
                fp.write(
                    "f %d/%d/%d %d/%d/%d %d/%d/%d\n"
                    % (
                        f[0] + faceIdOffset,
                        a_uvsIDs[current_f, 0] + uvsIdOffset,
                        a_normalsIDs[current_f, 0] + normalsIdOffset,
                        f[1] + faceIdOffset,
                        a_uvsIDs[current_f, 1] + uvsIdOffset,
                        a_normalsIDs[current_f, 1] + normalsIdOffset,
                        f[2] + faceIdOffset,
                        a_uvsIDs[current_f, 2] + uvsIdOffset,
                        a_normalsIDs[current_f, 2] + normalsIdOffset,
                    )
                )
            current_f += 1
    # print("Obj written in " + a_pathToSave)


class ConvergenceStatus(enum.Enum):
    TARGET_REACHED = enum.auto()
    STUCK = enum.auto()
    MAX_ITERATIONS_REACHED = enum.auto()


def optimize_avatar(
    avatar: Avatar, target_vertices: torch.tensor, max_iters=1000, learn_rate_min=0.05
):
    learn_rate = 0.2
    current_iters = 0
    consecutive_iters_checkpoint = 20
    trans, pose, shape = avatar.get_params()
    shape.requires_grad = True
    finished = False
    convergence_status = ConvergenceStatus.STUCK
    debug_anomalies = False  # True if you want to analyze why NaN is appearing (optim slower)

    # Iterate trying to minimize the cost function so the avatar measurements get closer to the
    # target vertices
    optimizer = torch.optim.Adam([shape], lr=learn_rate, weight_decay=2e-5)
    checkpoint_error = None
    best_error = 1e20

    for i in range(0, max_iters):
        current_iters += 1
        optimizer.zero_grad()
        current_avatar_vertices = avatar.get_vertices_from_params(trans, pose, shape)
        # torch.dist is the same as calling (current_avatar_vertices - target_vertices).pow(2).sum().sqrt()
        # loss = torch.dist(current_avatar_vertices, target_vertices)
        # We will use squared distance since it's cheaper to calculate and penalizes
        # more vertices that are too far away.
        loss = (current_avatar_vertices - target_vertices).pow(2).sum()
        print(f"Loss: {loss}")
        if loss < 0.005:
            finished = True

        best_error = min(best_error, loss.item())
        if finished:
            convergence_status = ConvergenceStatus.TARGET_REACHED
            break
        # If the error doesn't improve after a while it means it's stuck so if possible we need
        # to try with another optimization updating the learning rate and resetting the weights.
        if i % consecutive_iters_checkpoint == 0:
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
    if current_iters == max_iters:
        convergence_status = ConvergenceStatus.MAX_ITERATIONS_REACHED
    if torch.isnan(shape).any():
        raise (
            ValueError,
            f"Something wrong with the shape after the optimization, found NaN on it {shape}",
        )
    avatar.set_params(shape=shape.detach())
    return convergence_status


def get_random_shape_params():
    # Build random shape parameters.
    shape_mean = torch.zeros((1, 300))
    shape_stdev = torch.full((1, 300), 2.0)
    shape = torch.normal(mean=shape_mean, std=shape_stdev)
    return shape


if __name__ == "__main__":
    # # -------- Params
    gender = "male"
    avatar_type = AvatarType.SMPLX
    output_dir = Path(
        "D:\-DYDDV - URJC\SEDDI\Git\smplx-deca\Demos"
    )
    # --------
    target_avatar = Avatar(gender, avatar_type, shape=get_random_shape_params())
    # We need to detach the target vertices to avoid the require_gradient that come by default
    # from SMPLX (due to the concatenation with betas and expression).
    target_vertices = target_avatar.get_vertices()
    # print(f"Target_vertices for {avatar_type.name} require grad: {str(target_vertices.requires_grad)}")
    avatar = Avatar(gender, avatar_type)
    write_obj(
        output_dir / "starting_avatar.obj",
        avatar.get_vertices().detach().numpy(),
        avatar.get_faces(),
    )
    write_obj(
        output_dir / "target_avatar.obj",
        target_vertices.detach().numpy(),
        target_avatar.get_faces(),
    )
    exit_reason = optimize_avatar(avatar, target_vertices)
    write_obj(
        output_dir / "optimized_avatar.obj",
        avatar.get_vertices().detach().numpy(),
        avatar.get_faces(),
    )
    print(f"Optimization finished, exit condition: {exit_reason.name}")

