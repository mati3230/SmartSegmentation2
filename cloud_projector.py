import numpy as np
from utils import gl_frustum,\
    gl_perspective,\
    get_rotation_mat,\
    grad_to_rad,\
    PtoImg2,\
    get_bounding_box


class CloudProjector:
    def __init__(
            self,
            state_size):
        self.width = int(state_size[1])
        height = int(state_size[2])
        self.imgs = np.zeros(state_size)
        self.aspect_ratio = self.width / height
        angle_of_view = 90
        near = 0.1
        far = 100
        right, left, top, bottom = gl_perspective(
            angle_of_view=angle_of_view,
            aspect_ratio=self.aspect_ratio,
            n=near)
        self.M_proj = gl_frustum(
            bottom,
            top,
            left,
            right,
            near,
            far)
        self.angle_1 = grad_to_rad(-30)
        self.angle_2 = grad_to_rad(30)
        # around x-axis
        self.axis_1 = np.array([0, 1, 0])
        # aorund y-axis
        self.axis_2 = np.array([1, 0, 0])
        self.distance = 6
        self.dist_factor = 2.5
        self.r1 = get_rotation_mat(
            angle=self.angle_1,
            axis=self.axis_1)
        self.r2 = get_rotation_mat(
            angle=self.angle_1,
            axis=self.axis_2)
        self.r3 = get_rotation_mat(
            angle=self.angle_2,
            axis=self.axis_1)
        self.r4 = get_rotation_mat(
            angle=self.angle_2,
            axis=self.axis_2)

    def reset_for_img(self, t):
        Rt = np.eye(4)
        Rt = np.tile(Rt, (self.imgs.shape[0], 1, 1))
        Rt[0, :3, :3] = self.r1
        Rt[0, :3, 3] = t[:, 0]
        Rt[1, :3, :3] = self.r2
        Rt[1, :3, 3] = t[:, 1]
        Rt[2, :3, :3] = self.r3
        Rt[2, :3, 3] = t[:, 2]
        Rt[3, :3, :3] = self.r4
        Rt[3, :3, 3] = t[:, 3]
        return Rt

    def preprocess(self, state):
        P = np.array(state[0][:, :3])
        C = np.array(state[0][:, 3:])
        # transform color values in the range of 0 to 1
        C /= 255
        P_idxs = state[1]
        neighbour_idxs = state[2]
        if P_idxs.shape[0] == 0 and neighbour_idxs.shape[0] == 0:
            camera_translation = np.array([0, 0, 0])
        else:
            # color the main superpoint in red
            C[P_idxs, 2] = 1
            # lower other color values of the main superpoint
            C[P_idxs, :2] /= 4

            # color the neighbour superpoint in blue
            C[neighbour_idxs, 0] = 1
            # lower other color values of the neighbour superpoint
            C[neighbour_idxs, 1:] /= 4

            # calculate camera_translation of segments
            main_idxs = np.vstack(
                (P_idxs[:, None], neighbour_idxs[:, None])).reshape(-1)
            P_main = P[main_idxs]
            bb_main = get_bounding_box(P_main)
            l1 = bb_main[1] - bb_main[0]
            l2 = bb_main[3] - bb_main[2]

            l = max(l1, l2)
            h = self.dist_factor * l + self.distance

            # x direction and up
            ct1 = np.array([0, -l2 / 2, h])
            # y direction and up
            ct2 = np.array([-l1 / 2, 0, h])
            ct3 = np.array([0, l2 / 2, h])
            ct4 = np.array([l1 / 2, 0, h])
            v1 = np.array([-bb_main[0], -bb_main[2], bb_main[5]])
            v2 = np.array([-bb_main[1], -bb_main[3], bb_main[5]])
            p1 = (v1 + ct1)[:, None]
            p2 = (v1 + ct2)[:, None]
            p3 = (v2 + ct3)[:, None]
            p4 = (v2 + ct4)[:, None]
            camera_translation = np.hstack((p1, p2, p3, p4))

        # transform color values in the range of -1 to 1
        C -= 0.5
        C *= 2

        P = np.transpose(P)
        P = np.vstack((P, np.ones((1, P.shape[1]))))

        Rt = self.reset_for_img(camera_translation)
        self.imgs[:] = 0
        res, mask, idxs = PtoImg2(P, self.M_proj, Rt, self.width)
        idx_start = 0
        for i in range(self.imgs.shape[0]):
            idx = idxs[i]
            idx_split = idx_start + idx
            x = res[idx_start:idx_split]
            idx_start = idx_split + idx
            y = res[idx_split:idx_start]
            m = mask[i]
            c = C[m, :]
            self.imgs[i, x, y] = c
        return self.imgs
