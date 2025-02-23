import pytorch3d.loss
import pytorch3d.ops
import torch, pytorch3d

# define losses
def voxel_loss(voxel_src, voxel_tgt, reduction='mean'):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grids
	voxel_src = torch.clamp(voxel_src, min=0.0, max=1.0)
	criterian = torch.nn.BCELoss(reduction=reduction)
	loss = criterian(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src, point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	B1, N1, _ = point_cloud_src.shape
	B2, N2, _ = point_cloud_tgt.shape
	assert B1==B2, "B1 != B2"
	chamfer_dist_src, _, _ = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, norm=2, K=1)
	chamfer_dist_tgt, _, _ = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, norm=2, K=1)
	chamfer_loss_src = torch.sum(chamfer_dist_src.reshape(-1))
	chamfer_loss_tgt = torch.sum(chamfer_dist_tgt.reshape(-1))
	loss_chamfer = chamfer_loss_src/(2*N1)+chamfer_loss_tgt/(2*N2)
	return loss_chamfer

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss
	loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian
