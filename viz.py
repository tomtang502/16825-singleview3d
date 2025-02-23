import torch, imageio, math, mcubes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PointLights,
    RasterizationSettings,
    HardPhongShader,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.structures import Meshes, Pointclouds
import numpy as np
import matplotlib as mpl
from rend_utils import get_points_renderer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.dpi'] = 120

def view_360(mesh, device, num_frames, elev, dist, image_size):
    angles = torch.linspace(0, 2*torch.pi, num_frames)
    distances = [dist]*num_frames
    elevs = [elev]*num_frames
    R, t = look_at_view_transform(distances, elevs, angles, degrees=False)
    cameras = FoVPerspectiveCameras(
        R=R.to(device), T=t.to(device), fov=60, device=device
    )
    lights = PointLights(location=[[0, 0, -3]], device=device)
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )
    rendered_imgs = renderer(mesh.extend(num_frames))
    return rendered_imgs

def pc_view360(pc, device, num_frames, elev, dist, image_size=(480, 640), radius=0.01):
    points_renderer = get_points_renderer(
        image_size=image_size,
        radius=radius,
    )
    angles = torch.linspace(0, 2*torch.pi, num_frames)
    distances = [dist]*num_frames
    elevs = [elev]*num_frames
    R, t = look_at_view_transform(distances, elevs, angles, degrees=False)
    cameras = FoVPerspectiveCameras(
        R=R.to(device), T=t.to(device), fov=60, device=device
    )
    return points_renderer(pc.to(device).extend(num_frames), cameras=cameras)

def mesh_view360_gif(mesh, save_path, device, add_texture=False, num_frames=100, elev=0.5, dist=2, duration=1.0, image_size=(480, 640)):
    if add_texture:
        verts_list = mesh.verts_list()  # List of tensors, one per mesh
        faces_list = mesh.faces_list()  # List of tensors, one per mesh
        tex_list = []
        verts_detached_list = []
        for verts in verts_list:
            verts = verts.detach()
            verts_detached_list.append(verts)
            vmin, _ = verts.min(axis=0)
            vmax, _ = verts.max(axis=0)
            verts_normalized = (verts - vmin) / (vmax - vmin)
            tex_list.append(verts_normalized.clone())
        textures = TexturesVertex(verts_features=tex_list)
        mesh = Meshes(verts=verts_detached_list, faces=faces_list, textures=textures)
    rendered_imgs = view_360(mesh, device, num_frames, elev, dist, image_size)
    rendered_imgs_rgb = rendered_imgs[..., :3]
    rendered_imgs_rgb_L = [(rendered_imgs_rgb[i].cpu().numpy()*255).astype(np.uint8) for i in range(rendered_imgs_rgb.shape[0])]
    duration = duration / num_frames
    imageio.mimsave(save_path, rendered_imgs_rgb_L, duration=duration, loop=0)

def pc_view360_gif(xyzs, save_path, device, rgbs=None, num_frames=100, elev=0.5, dist=2, duration=1.0, image_size=(480, 640), radius=0.01):
    if rgbs is None:
        rgbs = []
        verts_list = []
        for xyz in xyzs:
            verts = xyz.detach()
            verts_list.append(verts)
            vmin, _ = verts.min(axis=0)
            vmax, _ = verts.max(axis=0)
            verts_normalized = (verts - vmin) / (vmax - vmin)
            rgbs.append(verts_normalized)
        xyzs = verts_list
    pc = Pointclouds(points=xyzs, features=rgbs)  
    rendered_imgs = pc_view360(pc, device, num_frames, elev, dist, image_size=image_size, radius=radius)
    rendered_imgs_rgb = rendered_imgs[..., :3]
    rendered_imgs_rgb_L = [(rendered_imgs_rgb[i].cpu().numpy()*255).astype(np.uint8) for i in range(rendered_imgs_rgb.shape[0])]
    duration = duration / num_frames
    imageio.mimsave(save_path, rendered_imgs_rgb_L, duration=duration, loop=0)
    
def render_voxel_by_mesh(voxels, device, save_path, dist=6, image_size=(480, 640), norm_size=2):
    # voxels: tensor of b x h x w x d
    b, h, w, d = voxels.shape
    threshold = 0.5
    batch_verts, batch_faces, batch_tex = [], [], []
    voxel_numpy = voxels.detach().cpu().numpy()
    for i in range(b):
        volume = voxel_numpy[i]
        vertices, triangles = mcubes.marching_cubes(volume, isovalue=threshold)
        verts_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
        vmin, _ = verts_tensor.min(axis=0)
        vmax, _ = verts_tensor.max(axis=0)
        verts_normalized = (verts_tensor - vmin) / (vmax - vmin)
        textures = verts_normalized.clone()
        
        verts_normalized = verts_normalized*2*norm_size-norm_size
        
        faces_tensor = torch.tensor(triangles, dtype=torch.int64, device=device)
        batch_verts.append(verts_normalized)
        batch_faces.append(faces_tensor)
        batch_tex.append(textures)
    # batch_verts = torch.stack(batch_verts)
    # batch_faces = torch.stack(batch_faces)
    mesh = Meshes(verts=batch_verts, faces=batch_faces, textures=TexturesVertex(batch_tex))
    mesh_view360_gif(mesh, save_path, device, num_frames=100, elev=0.3, dist=dist, duration=1.0, image_size=image_size)


# def render_voxel_batch(voxel_batch, batch_index=0, threshold=0.5, cmap='viridis', save_path='voxel_plot.png'):
   
#     voxel = voxel_batch[batch_index]
#     occupied = voxel > threshold

#     norm_voxel = (voxel - voxel.min()) / (voxel.max() - voxel.min() + 1e-8)
#     facecolors = plt.cm.get_cmap(cmap)(norm_voxel)
#     facecolors[~occupied] = (0, 0, 0, 0)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.voxels(occupied, facecolors=facecolors, edgecolor='k')
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.savefig(save_path)
    
#     plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    image_size = (480, 640)
    save_dir = "output/explore_loss"
    
    voxel_data_path = f"{save_dir}/fitted_voxels.pth"
    voxel_dict = torch.load(voxel_data_path)
    voxel_src_save_path = f"{save_dir}/viz_voxel_src.gif"
    voxel_tgt_save_path = f"{save_dir}/viz_voxel_tgt.gif"
    render_voxel_by_mesh(voxel_dict['src'], device, voxel_src_save_path, image_size)
    render_voxel_by_mesh(voxel_dict['tgt'], device, voxel_tgt_save_path, image_size)
    
    mesh_data_path = f"{save_dir}/fitted_meshes.pth"
    mesh_dict = torch.load(mesh_data_path)
    mesh_src_save_path = f"{save_dir}/viz_mesh_src.gif"
    mesh_tgt_save_path = f"{save_dir}/viz_mesh_tgt.gif"
    mesh_view360_gif(mesh_dict['src'], mesh_src_save_path, device, add_texture=True, image_size=image_size)
    mesh_view360_gif(mesh_dict['tgt'], mesh_tgt_save_path, device, add_texture=True, image_size=image_size)
    
    point_data_path = f"{save_dir}/fitted_pointcloud.pth"
    point_dict = torch.load(point_data_path)
    point_src_save_path = f"{save_dir}/viz_point_src.gif"
    point_tgt_save_path = f"{save_dir}/viz_point_tgt.gif"
    pc_view360_gif(point_dict['src'], save_path=point_src_save_path, device=device, image_size=image_size)
    pc_view360_gif(point_dict['tgt'], save_path=point_tgt_save_path, device=device, image_size=image_size)