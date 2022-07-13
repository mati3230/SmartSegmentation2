import numpy as np
from scipy.spatial.transform import Rotation as Rot
import tensorflow as tf
# import cv2
import time


def get_bounding_box(P):
    """Get the bounding box of a point cloud P.

    Parameters
    ----------
    P : np.ndarray
        Point cloud.

    Returns
    -------
    tuple(float, float, float, float, float, float)
        Bounding box.

    """
    return (
            np.min(P[:, 0]),
            np.max(P[:, 0]),
            np.min(P[:, 1]),
            np.max(P[:, 1]),
            np.min(P[:, 2]),
            np.max(P[:, 2]))


def get_rotation_mat(angle, axis):
    """Returns the rotation matrix from an angle and axis.

    Parameters
    ----------
    angle : float
        Angle of the rotation.
    axis : int
        Axis (x, y, z) of the rotation.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.

    """
    r = Rot.from_rotvec(angle * axis)
    return r.as_matrix()


def gl_perspective(angle_of_view, aspect_ratio, n):
    """Set up the variables that are necessary to compute the perspective
    projection matrix. See also:
    https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml

    Parameters
    ----------
    angle_of_view : float
         Specifies the field of view angle, in degrees, in the y direction.
    aspect_ratio : float
        Specifies the aspect ratio that determines the field of view in the x
        direction. The aspect ratio is the ratio of x (width) to y (height).
    n : float
        Specifies the distance from the viewer to the near clipping plane
        (always positive).

    Returns
    -------
    tuple(float, float, float, float)
        Right, left, top, bottom.

    """
    scale = np.tan(angle_of_view * 0.5 * np.pi / 180) * n
    r = aspect_ratio * scale
    l = -r
    t = scale
    b = -t
    return r, l, t, b


def gl_frustum(b, t, l, r, n, f):
    """Set up the perspective projection matrix.

    Parameters
    ----------
    b : float
        Bottom.
    t : float
        Top.
    l : float
        Left.
    r : float
        Left.
    n : float
        Distance of the near plane.
    f : float
        Distance of the far plane.

    Returns
    -------
    np.ndarray
        Perspective projection matrix..

    """
    M = np.zeros((4, 4))
    M[0, 0] = 2 * n / (r - l)
    M[1, 1] = 2 * n / (t - b)

    M[2, 0] = (r + l) / (r - l)
    M[2, 1] = (t + b) / (t - b)
    M[2, 2] = -(f + n) / (f - n)
    M[2, 3] = -1

    M[3, 2] = -2 * f * n / (f - n)
    return M


def mult_points_matrix(M, P):
    """Multiply the point cloud by a transformation matrix M.

    Parameters
    ----------
    M : np.ndarray
        Transformation matrix.
    P : np.ndarray
        Point cloud.

    Returns
    -------
    np.ndarray
        Transformed point cloud.

    """
    P_ = np.matmul(M, P)
    P_[:3, :] = P_[:3, :] / P_[3, :]
    return P_


def PtoImg(
        P,
        C,
        M_proj,
        aspect_ratio,
        width,
        height,
        t,
        rotation_mat=np.eye(3)):
    """Perspective projection of a point cloud to an image.

    Parameters
    ----------
    P : np.ndarray
        Point cloud (X, Y, Z coordinates only).
    C : np.ndarray
        Color matrix of the points in the form nx3 where n is the number of
        points.
    M_proj : np.ndarray
        Projection matrix.
    aspect_ratio : float
        Description of parameter `aspect_ratio`.
    width : int
        Width of the output image.
    height : int
        Height of the output image.
    t : np.ndarray
        Translation of the camera.
    rotation_mat : np.ndarray
        Rotation matrix to rotate the camera.

    Returns
    -------
    np.ndarray
        Rendered image of the point cloud with a perspective camera.

    """
    img = np.zeros((width, height, 3))

    P = np.vstack((P, np.ones((1, P.shape[1]))))

    Rt = np.eye(4)
    Rt[:3, :3] = rotation_mat
    Rt[:3, 3] = t
    # P = np.matmul(Rt, P)

    M_proj = np.matmul(M_proj, Rt)

    P_proj = mult_points_matrix(M_proj, P)
    P_proj = P_proj[:2, :]

    in_image_indxs = np.where(
        (P_proj[0, :] > -aspect_ratio) &
        (P_proj[0, :] < aspect_ratio) &
        (P_proj[1, :] > -1) &
        (P_proj[1, :] < 1))[0]
    if in_image_indxs.shape[0] == 0:
        return img

    P_proj = P_proj[:, in_image_indxs]
    min1 = np.ones((1, P_proj.shape[1]))
    min1[0, :] = width - 1
    P_proj[0, :] = np.minimum(min1, (P_proj[0, :] + 1) * 0.5 * width)

    min1[0, :] = height - 1
    P_proj[1, :] = np.minimum(min1, (P_proj[1, :] + 1) * 0.5 * height)

    proj1 = P_proj.astype(np.int32)

    idxs = np.arange(P.shape[1])
    idxs = idxs[:, None]
    _, idxs_intersect, P_in_image_indxs_intersect = np.intersect1d(
        idxs, in_image_indxs, return_indices=True)

    img[proj1[0, P_in_image_indxs_intersect],
        proj1[1, P_in_image_indxs_intersect], :] =\
        C[idxs_intersect]

    return img


def PtoImg1(
        img,
        P,
        C,
        M_proj,
        aspect_ratio,
        width,
        height,
        Rt):
    """Perspective projection of a point cloud to an image.

    Parameters
    ----------
    img : np.ndarray
        The image that should be returned.
    P : np.ndarray
        Point cloud (X, Y, Z coordinates only).
    C : np.ndarray
        Color matrix of the points in the form nx3 where n is the number of
        points.
    M_proj : np.ndarray
        Projection matrix.
    aspect_ratio : float
        Description of parameter `aspect_ratio`.
    width : int
        Width of the output image.
    height : int
        Height of the output image.
    Rt : np.ndarray
        Transformation matrix to transform the camera (rotation and
        translation).

    Returns
    -------
    np.ndarray
        Rendered image of the point cloud with a perspective camera.

    """
    # transform points into camera space
    M_proj = np.matmul(M_proj, Rt)

    P_proj = mult_points_matrix(M_proj, P)
    # points of the transformed cloud in image space
    P_proj = P_proj[:2, :]

    # idxs of the points of the transformed cloud that are in the image
    in_image_indxs = np.where(
        (P_proj[0, :] > -aspect_ratio) &
        (P_proj[0, :] < aspect_ratio) &
        (P_proj[1, :] > -1) &
        (P_proj[1, :] < 1))[0]
    # if no point is in the image
    if in_image_indxs.shape[0] == 0:
        return img

    # filter points that will not be in the image
    P_proj = P_proj[:, in_image_indxs]
    # scale to pixel range (e.g. [-1, 1] -> [0, width])
    P_proj[0, :] = (P_proj[0, :] + 1) * 0.5 * width
    P_proj[1, :] = (P_proj[1, :] + 1) * 0.5 * height
    P_proj = P_proj.astype(np.int32)

    img[P_proj[0, :],
        P_proj[1, :], :] =\
        C[in_image_indxs]

    return img


@tf.function(
    input_signature=(
        tf.TensorSpec([None, 4, 4], dtype=tf.float32),
        tf.TensorSpec([4, None], dtype=tf.float32)
        )
)
def mult_points_matrix1(M, P):
    """Projects points from 3D to 2D

    Parameters
    ----------
    M : tf.Tensor
        Perspective projection matrix (4 X 4) to project points from 3D to 2D.
    P : np.ndarray
        Point cloud (B X 4 X |P|).

    Returns
    -------
    tf.Tensor
        Project point cloud (B X 3 X |P|).

    """
    # M = tf.reshape(M, (None, M.shape[0], M.shape[1]))
    P_ = tf.matmul(M[:, :], P[:])
    # print(P_.shape)
    D = P_[:, 3, :]
    # print((P_[:, :3, :] / D[:, None, :]).shape)
    return P_[:, :3, :] / D[:, None, :]


@tf.function(
    input_signature=(
        tf.TensorSpec([4, None], dtype=tf.float32),
        tf.TensorSpec([4, 4], dtype=tf.float32),
        tf.TensorSpec([None, 4, 4], dtype=tf.float32),
        tf.TensorSpec([], dtype=tf.int32)
        )
)
def PtoImg2(P, M_proj, Rt, wh):
    """Perspective projection of a point cloud to multiple images. The batch
    size determines the number of images.

    Parameters
    ----------
    P : np.ndarray
        Point clouds.
    M_proj : np.ndarray
        Projection Matrices.
    Rt : np.ndarray
        Matrices to rotate and translate the points.
    wh : int
        Dimension of the output images in pixel (e.g. 256).

    Returns
    -------
    list
        Description of returned object.
    tf.Tensor
        Description of returned object.

    """
    # transform points into camera space
    # 4 X 4
    M = M_proj
    M = tf.matmul(M, Rt)
    # print(M.shape)
    # B X 3 X |P|
    P_proj = mult_points_matrix1(M, P)
    # print(P_proj.shape)
    # points of the transformed cloud in image space
    # B X 2 X |P|
    P_proj = P_proj[:, :2, :]
    # print(P_proj.shape)

    # idxs of the points of the transformed cloud that are in the image
    # B X |P|
    wh_f = tf.cast(wh, tf.float32)
    # print(P_proj.shape)
    P_proj = (P_proj + 1) * 0.5 * wh_f
    P_proj = tf.cast(P_proj, tf.int32)
    mask1 = tf.where(
        (P_proj[:, 0, :] > 0) &
        (P_proj[:, 0, :] < wh) &
        (P_proj[:, 1, :] > 0) &
        (P_proj[:, 1, :] < wh),
        tf.ones_like(P_proj[:, 0], tf.bool),
        tf.zeros_like(P_proj[:, 0], tf.bool)
        )

    # print(P_proj.shape)
    # print(P_proj.shape)
    # filtered = []
    n_iter = tf.shape(Rt)[0]
    filtered = tf.TensorArray(tf.int32, size=n_iter)
    idxs = tf.TensorArray(tf.int32, size=n_iter)
    for i in tf.range(n_iter):
        P_proj_x = P_proj[i, 0, :]
        x = tf.boolean_mask(P_proj_x, mask1[i])
        P_proj_y = P_proj[i, 1, :]
        y = tf.boolean_mask(P_proj_y, mask1[i])
        # v = tf.stack([x, y])
        v = tf.concat([x, y], axis=0)
        filtered = filtered.write(i, v)
        idxs = idxs.write(i, tf.shape(x)[0])
    return filtered.concat(), mask1, idxs.stack()


def grad_to_rad(angle):
    """Transforms an angle from degrees to radians.

    Parameters
    ----------
    angle : float
        Angle in degrees.

    Returns
    -------
    float
        Transforms an angle from degrees to radians.

    """
    return (angle / 180) * np.pi


def cpu_vs_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    state_size = (4, 128, 128, 3)
    imgs = np.zeros(state_size)
    n_images = int(state_size[0])
    width = int(state_size[1])
    height = int(state_size[2])

    angle_of_view = 90
    near = 0.1
    far = 100
    aspect_ratio = width / height

    r, l, t, b = gl_perspective(
        angle_of_view=angle_of_view,
        aspect_ratio=aspect_ratio,
        n=near)
    M_proj = gl_frustum(b=b, t=t, l=l, r=r, n=near, f=far)

    angle_1 = grad_to_rad(-150)
    angle_2 = grad_to_rad(-210)
    axis_1 = np.array([1, 0, 0])
    axis_2 = np.array([0, 1, 0])
    distance = 6
    dist_factor = 2.5
    Rt = np.eye(4)
    Rt = np.tile(Rt, (n_images, 1, 1))
    r1 = get_rotation_mat(
        angle=angle_1,
        axis=axis_1)
    r2 = get_rotation_mat(
        angle=angle_2,
        axis=axis_1)
    r3 = get_rotation_mat(
        angle=angle_1,
        axis=axis_2)
    r4 = get_rotation_mat(
        angle=angle_2,
        axis=axis_2)
    # wh = np.array([width, height], np.float64)
    wh = 128
    t = np.array([0, 0, 10])
    Rt[0, :3, :3] = r1
    Rt[0, :3, 3] = t
    Rt[1, :3, :3] = r2
    Rt[1, :3, 3] = t
    Rt[2, :3, :3] = r3
    Rt[2, :3, 3] = t
    Rt[3, :3, :3] = r4
    Rt[3, :3, 3] = t

    n_points = 50000
    P = np.random.rand(3, n_points)
    P = np.vstack((P, np.ones((1, P.shape[1]))))
    C = np.random.rand(n_points, 3)
    """
    res, mask = PtoImg2(P, M_proj, Rt, wh)
    for i in range(n_images):
        r = res[i]
        x = r[0, :]
        y = r[1, :]
        m = mask[i]
        c = C[:, m].transpose()
        imgs[i, x, y] = c
    img_plot1 = np.concatenate((imgs[0], imgs[1]), axis=1)
    img_plot2 = np.concatenate((imgs[2], imgs[3]), axis=1)
    img_plot = np.concatenate((img_plot1, img_plot2), axis=0)

    cv2.imshow("img", img_plot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    img1 = np.zeros((state_size[1], state_size[2], state_size[3]))
    img2 = np.zeros((state_size[1], state_size[2], state_size[3]))
    img3 = np.zeros((state_size[1], state_size[2], state_size[3]))
    img4 = np.zeros((state_size[1], state_size[2], state_size[3]))
    test_steps = 3
    cpu = np.zeros((test_steps, ))
    gpu = np.zeros((test_steps, ))
    """
    for i in range(test_steps):
        t1 = time.time()
        imgs[:] = 0
        img1 = PtoImg1(
            img=img1,
            P=P,
            C=C,
            M_proj=M_proj,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            Rt=Rt[0])
        img2 = PtoImg1(
            img=img2,
            P=P,
            C=C,
            M_proj=M_proj,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            Rt=Rt[1])
        img3 = PtoImg1(
            img=img3,
            P=P,
            C=C,
            M_proj=M_proj,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            Rt=Rt[2])
        img4 = PtoImg1(
            img=img4,
            P=P,
            C=C,
            M_proj=M_proj,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            Rt=Rt[3])

        imgs[0] = img1
        imgs[1] = img2
        imgs[2] = img3
        imgs[3] = img4
        t2 = time.time()
        cpu[i] = t2 - t1
    """
    for j in range(test_steps):
        t1 = time.time()
        imgs[:] = 0
        res, mask, idxs = PtoImg2(P, M_proj, Rt, wh)
        idx_start = 0
        for i in range(n_images):
            idx = idxs[i]
            idx_split = idx_start + idx
            x = res[idx_start:idx_split]
            idx_start = idx_split + idx
            y = res[idx_split:idx_start]
            m = mask[i]
            c = C[m, :]
            imgs[i, x, y] = c
        t2 = time.time()
        gpu[j] = t2 - t1
    print("CPU:", np.mean(cpu[2:]), np.std(cpu[2:]))
    print("GPU:", np.mean(gpu[2:]), np.std(gpu[2:]))


def test_rendering():
    import open3d as o3d
    import cv2
    mesh = o3d.io.read_triangle_mesh("../PointNetTransfer/sn000000.ply")
    #o3d.visualization.draw_geometries([mesh])
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)

    bgr = np.zeros(rgb.shape)
    bgr[:, 0] = rgb[:, 2]
    bgr[:, 1] = rgb[:, 1]
    bgr[:, 2] = rgb[:, 0]

    xyz_mean = np.mean(xyz, axis=0)
    xyz = xyz - xyz_mean
    #rgb *= 255

    xyz = np.transpose(xyz)

    width = 512
    height = 512
    angle_of_view = 140
    near = 0.1
    far = 100
    aspect_ratio = width / height

    r, l, t, b = gl_perspective(
        angle_of_view=angle_of_view,
        aspect_ratio=aspect_ratio,
        n=near)
    M_proj = gl_frustum(b=b, t=t, l=l, r=r, n=near, f=far)

    angle = grad_to_rad(90)
    axis = np.array([0, 0, -1])

    R = get_rotation_mat(
        angle=angle,
        axis=axis)
    t=np.array([0,0,-9])

    img = PtoImg(
        P=xyz,
        C=bgr,
        M_proj=M_proj,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        t=t,
        rotation_mat=R)

    cv2.imshow("test", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    #print(img[img != 0])
    img *= 255
    img = np.floor(img)
    img = img.astype(np.uint8)
    cv2.imwrite("./sn000000_rendering.jpg", img)


if __name__ == "__main__":
    test_rendering()