from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch, math
from pytorch3d.utils import ico_sphere
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes

def get_deconv2d(in_chann, out_chann, input_size, output_size, 
                       max_stride=10, max_kernel=10, max_padding=10):
    valid_params = []
    # Iterate over typical ranges. For output_padding, in PyTorch it must be less than stride.
    for stride in range(1, max_stride + 1):
        for kernel in range(1, max_kernel + 1):
            for padding in range(0, max_padding + 1):
                    # Compute output using the transposed convolution formula:
                    # O = (I-1)*stride - 2*padding + kernel_size + output_padding
                    o = (input_size - 1) * stride - 2 * padding + kernel
                    if o == output_size:
                        print(f"get dconv2d with K={kernel}, S={stride}, P={padding} to go from {input_size} to {output_size}...")
                        return nn.ConvTranspose2d(in_chann, out_chann, kernel_size=kernel, stride=stride, padding=padding)
    raise ValueError(f"No Valid param given {input_size} to {output_size} for deconv2d")

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            if args.type == "point":
                vision_model_L = list(vision_model.children())
                self.encoder_1 = torch.nn.Sequential(*(vision_model_L)[:6])
                self.encoder_2 = vision_model_L[6]
                self.encoder_3 = vision_model_L[7]
                self.avg_pool = vision_model_L[8]
                
            else:
                self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # idea from Pix2Vox https://arxiv.org/pdf/1901.11153
            # Input: b x 512 x 1 x 1 x 1
            # Output: b x 32 x 32 x 32
            # self.inter = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            self.dconv3d_block1 = nn.Sequential(
                nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True)
            )
            self.dconv3d_block2 = nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True)
            )
            
            self.dconv3d_block3 = nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            )
            
            self.dconv3d_block4 =nn.Sequential(
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            )
            
            self.dconv3d_block5 =nn.Sequential(
                nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True)
            )
            
            self.fdconv3d =nn.Sequential(
                nn.ConvTranspose3d(8, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
            
            self.decoder = nn.Sequential(
                self.dconv3d_block1,
                self.dconv3d_block2,
                self.dconv3d_block3,
                self.dconv3d_block4,
                self.dconv3d_block5,
                self.fdconv3d
            )            
        elif args.type == "point":
            # idea from https://arxiv.org/pdf/1612.00603 PSGN
            # Input: b x 512 x 1 x 1
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.n_sqrt = int(math.ceil(math.sqrt(self.n_point)))
            # handle residual connection preprocess
            self.conv_x0 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            self.conv_x1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
            self.conv_x2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1) # 256 5 5
            
            # decoder
            self.dconv0 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1, padding=0) # 256, 5, 5
            self.relu0 = nn.ReLU(inplace=True)
            
            self.decoder_conv2d_relu_0 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)   
            )
            
            self.dconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1) # 128, 9, 9
            self.relu1 = nn.ReLU(inplace=True)
            self.decoder_conv2d_relu_1 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)   
            )
            
            self.dconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 64, 18, 18
            self.relu2 = nn.ReLU(inplace=True)
            self.decoder_conv2d_relu_2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)   
            )
          
            self.dconv_relu = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32, 36, 36
                nn.ReLU(inplace=True),
                get_deconv2d(32, 16, input_size=36, output_size=self.n_sqrt),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
            )
                       
        elif args.type == "mesh":
            # idea from AtlasNet https://arxiv.org/pdf/1802.05384, except now deform a sphere
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.conv1 = nn.Conv1d(3, 512, 1)
            self.relu_bn1 = nn.Sequential(
                torch.nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            )
            self.fc_relu1 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
            )
            
            self.fc_relu2 = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(inplace=True)
            )
            
            self.conv_relu_bn = nn.Sequential(
                nn.Conv1d(2048, 512, 1),
                nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(512) 
            )
            
            self.conv5 = nn.Conv1d(512, 3, 1)
                        

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            if args.type == "point":
                x0 = self.encoder_1(images_normalize)
                x1 = self.encoder_2(x0)
                x2 = self.encoder_3(x1)
                encoded_feat = self.avg_pool(x2) # b x 512 x 1 x 1
            else:
                encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            encoded_feat_3d = encoded_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            voxels_pred = self.decoder(encoded_feat_3d).squeeze(1)        
            return voxels_pred
        elif args.type == "point":
            decoded_features = self.dconv0(encoded_feat)+self.conv_x2(x2)
            decoded_features = self.relu0(decoded_features)
            decoded_features = self.decoder_conv2d_relu_0(decoded_features)
            
            decoded_features = self.dconv1(decoded_features)+self.conv_x1(x1)
            decoded_features = self.relu1(decoded_features)
            decoded_features = self.decoder_conv2d_relu_1(decoded_features)
            
            decoded_features = self.dconv2(decoded_features)+self.conv_x0(x0)
            decoded_features = self.relu2(decoded_features)
            decoded_features = self.decoder_conv2d_relu_2(decoded_features)
            
            pointclouds_pred = self.dconv_relu(decoded_features)
            
            B, _, _, _ = pointclouds_pred.shape
            return pointclouds_pred.reshape(B, -1, 3)

        elif args.type == "mesh":
            encoded_feat = encoded_feat.unsqueeze(-1) # b x 512 x 1
            deform_vertices_pred = self.mesh_pred.verts_padded().permute(0, 2, 1)
            verts_embed = self.conv1(deform_vertices_pred)
            deform_hidden_pred = self.relu_bn1(verts_embed+encoded_feat)
            deform_hidden_pred = deform_hidden_pred.reshape(B, -1, 512)
            deform_hidden_pred = self.fc_relu1(deform_hidden_pred)
            deform_hidden_pred = self.fc_relu2(deform_hidden_pred)
            deform_hidden_pred = deform_hidden_pred.reshape(B, 2048, -1)
            deform_hidden_pred = self.conv_relu_bn(deform_hidden_pred)
            deform_vertices_pred = self.conv5(deform_hidden_pred)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return mesh_pred          

