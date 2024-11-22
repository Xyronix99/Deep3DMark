import os
import torch
import torchvision
from torch.functional import F
import numpy as np
import pyvista
import trimesh
import matplotlib.pyplot as plt
plt.switch_backend('agg')
pyvista.global_theme.background = 'white'

def read_mesh(path):
    _, ext = os.path.splitext(path)
    mesh = trimesh.load_mesh(path)
    xyz = torch.Tensor(np.asarray(mesh.vertices)).cuda().float().unsqueeze(0).contiguous()
    face = torch.Tensor(np.asarray(mesh.faces)).cuda().int().unsqueeze(0)
    return xyz, face

def write_mesh(path, pc, face):
    pc, face = to_numpy(pc), to_numpy(face)
    face = regulate_face(face)
    mesh = trimesh.Trimesh(pc, face)
    trimesh.exchange.export.export_mesh(mesh, path)

def write_msg(path, msg):
    np.savetxt(path, to_numpy(msg))

def to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.cpu().detach().numpy()
    return a

def regulate_face(face):
    """
        face: (N, 3)
    """
    for i, f in enumerate(face):
        if f[0]==-1:
            face = face[:i]
            break
    return face

def plot(x, y, path):
    fig, ax = plt.subplots()
    plt.xlabel('epoch')
    plt.ylabel('metrics')
    plt.plot(x, y)
    plt.grid(True)
    plt.savefig(path)

def visual_mesh(pc, face, idx=None):
    '''
        pc: (N, 3) Tensor or ndarray
        faces: (N, 3)
        idx: (N, nsample)
    '''
    pc, face = to_numpy(pc), to_numpy(face)
    face = regulate_face(face)
    N, _ = face.shape
    mesh = pyvista.PolyData(pc, face)
    ploter_args = dict(window_size=[1024, 768])
    p = pyvista.Plotter(**ploter_args)
    p.add_mesh(mesh, show_edges=False)
    p.show(screenshot=os.path.join("vis.png"))

def visual_pc(pc, idx = None):
    '''
        pc: (N, 3) Tensor or ndarray
        idx: (N, nsample)
    '''
    pc = to_numpy(pc)
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pc)
    )
    if idx!=None:
        idx = to_numpy(idx)
        N, nsample = idx.shape
        color = np.array([[128, 0, 0]] * N)
        selected_center = np.random.randint(0, N, size=(1,))
        neighbors = idx[selected_center]
        color[neighbors] = [0, 128, 0]
        color[selected_center] = [0, 0, 128]
        pcd.colors = o3d.utility.Vector3dVector(color)
        
    o3d.visualization.draw_geometries([pcd])
    return pcd

def save_images(original_images, watermarked_images, path, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    diff_image = images - watermarked_images
    torchvision.utils.save_image(stacked_images, os.path.join(path, "1.png"), normalize=False)
    torchvision.utils.save_image(diff_image, os.path.join(path, "diff.png"), normalize=False)

def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, path, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.capture_screen_image(path, True)
    vis.destroy_window()

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer(True)
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer(True)
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def render_mesh(mesh, path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(path, True)
    color = vis.capture_screen_float_buffer(True)
    depth = vis.capture_depth_float_buffer(True)

    vis.destroy_window()
    color = np.asarray(color)
    depth = np.asarray(depth)
    plt.imshow(color)
    plt.show()
    plt.imshow(depth)
    plt.show()

def visual_NDfeature(feature, path):
    """
        feature: (B, N, D)
    """
    os.makedirs(path, exist_ok=True)
    for i, feat in enumerate(feature):
        filepath = os.path.join(path, f'{i}.png')
        feat = (feat-feat.min())/(feat.max() - feat.min()) * 255
        cv2.imwrite(filepath, feat)

def plot_dist(dist, path):
    """
        dist: (N, )
    """
    dist = to_numpy(dist)
    fig, axs = plt.subplots(1, 1)
    _, _, bars = axs.hist(dist, log=True, bins=100)
    plt.bar_label(bars)
    plt.savefig(path)