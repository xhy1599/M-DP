#!/usr/bin/env python
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
import time
import math
import cv2
import matplotlib.pyplot as plt
import sys
import os
from cv_bridge import CvBridge
from threading import Lock
from PIL import Image as PILImage
import yaml


from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from pointnet import PointNet
from pointnet2_test import PointNet2ClsSsg
from dgcnn import DGCNN
from pointnetcov import PointConvDensityClsSsg

from downconv import PointNetSaModule

from mdptest.mdp_utils import load_model, to_numpy, unnormalize_data, from_numpy, transform_images





def PreProcess(PC_f1):    ####    pre process procedure

        batch_size = len(PC_f1)
        PC_f1_concat = []
        PC_f1_aft_aug = []

        for p in PC_f1:
            
            num_points = torch.tensor(p.shape[0], dtype=torch.int32)
            add_T = torch.ones(num_points, 1).cuda().to(torch.float32)
            PC_f1_add = torch.cat([p, add_T], -1)  ##  concat one to form   n 4
            PC_f1_concat.append(PC_f1_add)



        for i in range(batch_size):


            cur_PC_f1_concat = PC_f1_concat[i]
   
            ##  select the 20m * 20m region ########
            r_f1 = torch.norm(cur_PC_f1_concat[:, :2], p=2, dim =1, keepdim = True).repeat(1, 4)
            cur_PC_f1_concat = torch.where( r_f1 > 20 , torch.zeros_like(cur_PC_f1_concat).cuda(), cur_PC_f1_concat ).to(torch.float32)
            PC_mask_valid1 = torch.any(cur_PC_f1_concat != 0, dim=-1).cuda().detach()  # H W
            cur_PC_f1_concat = cur_PC_f1_concat[PC_mask_valid1 > 0,:]



            #####   generate  the  valid  mask (remove the not valid points)
            mask_valid_f1 = torch.any(cur_PC_f1_concat != 0, dim=-1, keepdim=True).cuda().detach()  # N 1
            mask_valid_f1 = mask_valid_f1.to(torch.float32)

           

            cur_PC_f1_concat = cur_PC_f1_concat[:, :3]
            cur_PC_f1_mask = cur_PC_f1_concat * mask_valid_f1 # N 3

            PC_f1_aft_aug.append(cur_PC_f1_mask)### list




        return PC_f1_aft_aug


def ProjectPCimg2SphericalRing(PC, Feature = None, H_input = 64, W_input = 1800):
 
    
    batch_size = len(PC)

    if Feature != None:
        num_channel = Feature[0].shape[-1]

    degree2radian = math.pi / 180
    nLines = H_input
    AzimuthResolution = 360.0 / W_input # degree
    VerticalViewDown = -35.0
    VerticalViewUp = 35.0

    # specifications of Velodyne-64
    AzimuthResolution = AzimuthResolution * degree2radian
    VerticalViewDown = VerticalViewDown * degree2radian
    VerticalViewUp = VerticalViewUp * degree2radian
    VerticalResolution = (VerticalViewUp - VerticalViewDown) / (nLines - 1)
    VerticalPixelsOffset = -VerticalViewDown / VerticalResolution

    # parameters for spherical ring's bounds

    PI = math.pi

    for batch_idx in range(batch_size):

        ###  initialize current processed frame

        cur_PC = PC[batch_idx].to(torch.float32)  # N  3
        # print(cur_PC.shape)
        if Feature != None:
            cur_Feature = Feature[batch_idx]  # N  c

        x = cur_PC[:, 0]
        y = cur_PC[:, 1]
        z = cur_PC[:, 2]

        r = torch.norm(cur_PC, p=2, dim =1)

        PC_project_current = torch.zeros([H_input, W_input, 3]).cuda().detach()  # shape H W 3
        if Feature != None:
            Feature_project_current = torch.zeros([H_input, W_input, num_channel]).cuda().detach()

        

        ####  get iCol & iRow

        iCol = ((PI -torch.atan2(y, x))/ AzimuthResolution) # alpha
        iCol = iCol.to(torch.int32)

        beta = torch.asin(z/r)                              # beta

        tmp_int = (beta / VerticalResolution + VerticalPixelsOffset)
        tmp_int = tmp_int.to(torch.int32)

        iRow = H_input - tmp_int

        iRow = torch.clamp(iRow, 0, H_input - 1)
        iCol = torch.clamp(iCol, 0, W_input - 1)

        iRow = iRow.to(torch.long)  # N 1
        iCol = iCol.to(torch.long)  # N 1


        PC_project_current[iRow, iCol, :] = cur_PC[:, :]  # H W 3
        if Feature != None:
            Feature_project_current[iRow, iCol, :] = cur_Feature[:, :]


        # Generate mask

        PC_mask_valid = torch.any(PC_project_current != 0, dim=-1).cuda().detach()  # H W
        PC_mask_valid = torch.unsqueeze(PC_mask_valid, dim=2).to(torch.float32) # H W 1

        if Feature != None:
            Feature_mask_valid = ~torch.any(Feature_project_current!= 0, dim=-1).cuda().detach()  # H W
            Feature_mask_valid = torch.unsqueeze(Feature_mask_valid, dim=2).to(torch.float32)

        ####1 h w
        PC_project_current = torch.unsqueeze(PC_project_current, dim=0)
        PC_mask_valid = torch.unsqueeze(PC_mask_valid, dim=0)


        if Feature != None:
            Feature_project_current = torch.unsqueeze(Feature_project_current,dim=0)
            Feature_mask_valid = torch.unsqueeze(Feature_mask_valid, dim=0)
        ####b h w
        if batch_idx == 0:
            PC_project_final = PC_project_current
            PC_mask_final = PC_mask_valid
            if Feature != None:
                Feature_project_final = Feature_project_current
                Feature_mask_final = Feature_mask_valid

        else:
            PC_project_final = torch.cat([PC_project_final, PC_project_current], 0)  # b h w 3
            PC_mask_final = torch.cat([PC_mask_final, PC_mask_valid], 0)
            if Feature != None:
                Feature_project_final = torch.cat([Feature_project_final, Feature_project_current], 0)
                Feature_mask_final = torch.cat([Feature_mask_final, Feature_mask_valid], 0)


    if Feature != None:
        return PC_project_final,  Feature_project_final
    else:
        return PC_project_final,  PC_mask_final



def get_selected_idx(batch_size, out_H: int, out_W: int, stride_H: int, stride_W: int):
    """According to given stride and output size, return the corresponding selected points

    Args:
        array (tf.Tensor): [any array with shape (B, H, W, 3)]
        stride_H (int): [stride in height]
        stride_W (int): [stride in width]
        out_H (int): [height of output array]
        out_W (int): [width of output array]
    Returns:
        [tf.Tensor]: [shape (B, outh, outw, 3) indices]
    """
    select_h_idx = torch.arange(0, out_H * stride_H, stride_H)
    select_w_idx = torch.arange(0, out_W * stride_W, stride_W)
    height_indices = (torch.reshape(select_h_idx, (1, -1, 1))).expand(batch_size, out_H, out_W)         # b out_H out_W
    width_indices = (torch.reshape(select_w_idx, (1, 1, -1))).expand(batch_size, out_H, out_W)            # b out_H out_W
    padding_indices = torch.reshape(torch.arange(batch_size), (-1, 1, 1)).expand(batch_size, out_H, out_W)   # b out_H out_W

    return padding_indices, height_indices, width_indices    



def visualize_pc_project(PC_project, mask=None, title="Cylindrical Projection"):
    """
    可视化圆柱投影后的点云图像。

    :param PC_project: 圆柱投影后的点云图像，形状为 [H_input, W_input, 3]
    :param mask: 掩码图像，形状为 [H_input, W_input, 1]，标记有效点（可选）
    :param title: 图像标题
    """
    # 检查输入形状
    if PC_project.ndim != 3 or PC_project.shape[2] != 3:
        raise ValueError("PC_project 的形状应为 [H_input, W_input, 3]")

    # 提取高度和宽度
    H_input, W_input, _ = PC_project.shape

    # 将 PC_project 转换为灰度图像（使用 z 坐标作为灰度值）
    gray_image = np.linalg.norm(PC_project, axis=2)  # 计算每个点的欧几里得范数
    gray_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image) + 1e-8)  # 归一化到 [0, 1]


   
 

    # 将灰度图像保存为 PNG 文件
    plt.figure(figsize=(10, 5))
    plt.imshow(gray_image, cmap='gray', extent=[0, W_input, H_input, 0])
    plt.colorbar(label="Z-coordinate (Normalized)")
    plt.title(title)
    plt.xlabel("Azimuth (Column Index)")
    plt.ylabel("Elevation (Row Index)")
    #plt.savefig('/home/dell/catkin_ws/src/realsense_pointnet_on_ROS-main/src/huiduzed.png', bbox_inches='tight', pad_inches=0, dpi=500)  # 保存图像
    plt.close()
    #plt.show(block=False)  # 非阻塞模式

    # 如果提供了掩码，可视化掩码
    if mask is not None:
        if mask.ndim != 3 or mask.shape[2] != 1:
            raise ValueError("mask 的形状应为 [H_input, W_input, 1]")
    
        plt.figure(figsize=(10, 5))
        plt.imshow(mask[:, :, 0], cmap='binary', extent=[0, W_input, H_input, 0])
        plt.title("Mask of Valid Points")
        plt.xlabel("Azimuth (Column Index)")
        plt.ylabel("Elevation (Row Index)")
        #plt.show(block=False)  # 非阻塞模式
        #plt.savefig('/home/dell/catkin_ws/src/realsense_pointnet_on_ROS-main/src/maskzed.png', bbox_inches='tight', pad_inches=0, dpi=500)  # 保存掩码图像
        plt.close()







class PointNetROS:
    def __init__(self):


        self.model = PointNet()


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.bridge = CvBridge()
        self.queue_lock = Lock()
        self.queue_lock1 = Lock()

        #####   initialize the parameters (distance  &  stride ) ######
        self.H_input=16
        self.W_input=1800
        
        self.Down_conv_dis = [0.75, 3.0, 6.0, 12.0]
        self.Up_conv_dis = [3.0, 6.0, 9.0]
        self.Cost_volume_dis = [1.0, 2.0, 4.5]
        
        self.stride_H_list = [4, 2, 2, 1]
        self.stride_W_list = [8, 2, 2, 2]

        self.out_H_list = [math.ceil(self.H_input / self.stride_H_list[0])]
        self.out_W_list = [math.ceil(self.W_input / self.stride_W_list[0])]

        for i in range(1, 4):
            self.out_H_list.append(math.ceil(self.out_H_list[i - 1] / self.stride_H_list[i])) ##(16,8,4,4)
            self.out_W_list.append(math.ceil(self.out_W_list[i - 1] / self.stride_W_list[i])) ##(57,29,15,8) # generate the output shape list
        
        


    
        self.w_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w_q = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)
        self.batch_size=1
        bn_decay=None
        self.training=False
        self.input_xyz_aug_proj_f1=None
        self.input_points_f1=None
        self.l0_xyz_proj_f1=None
        self.current_goal = None  





      



        # CONSTANTS

        MODEL_CONFIG_PATH = "/home/dell/catkin_ws/src/realsense_pointnet_on_ROS-main/src/mdptest/models.yaml"



        # Load the model 
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device:", self.device)




 


    # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        self.MAX_V = model_paths["max_v"]
        self.MAX_W = model_paths["max_w"]
        self.RATE = model_paths["frame_rate"] 
        self.context_size = None  
        self.context_queue=[]

        self.ACTION_STATS = {}
        for key in model_paths['action_stats']:
            self.ACTION_STATS[key] = np.array(model_paths['action_stats'][key])

        model_config_path = model_paths["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]

    # load model weights
        ckpth_path = model_paths["ckpt_path"]
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    
        self.model = load_model(
                ckpth_path,
                self.model_params,
                self.device,
            )
        self.model  = self.model.to(self.device)
        self.model.eval()

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
    
        print("model is ok")



        #self.point_cloud_sub = rospy.Subscriber("/zed2i/zed_node/point_cloud/cloud_registered", PointCloud2, self.callback)
        self.point_cloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.pc_callback, queue_size=10)
       # self.segmented_cloud_pub = rospy.Publisher("/segmented_cloud", PointCloud2, queue_size=10)
        self.image_sub = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, self.image_callback, queue_size=10, buff_size=2**24)
        self.goal_sub = rospy.Subscriber("goal", Path, self.goal_callback, queue_size=1)
        #self.semantic_sub = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, self.semantic_callback, queue_size=10, buff_size=2**24)
        

        



    def pc_callback(self, data):

        
        

        points = np.array([p[:3] for p in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)])
        points = points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)] 

        if points.size == 0:
            print("No valid points found in the point cloud")
            return
        points = torch.from_numpy(points).float()


        #points  = points.transpose(1, 2).contiguous()
        #print(points.shape)
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        points = points.clone().detach().to('cuda')


        pos1_list = []
        pos1_list.append(points)
        #pos1_list.append(points)
        
        #with torch.no_grad():
            
        #    predictions,_,_ = self.model(points)

        try:

            with self.queue_lock:
                

                input_xyz_aug_f1= PreProcess(pos1_list)
                input_xyz_aug_proj_f1, mask_xyz_f1 = ProjectPCimg2SphericalRing(input_xyz_aug_f1, None, self.H_input, self.W_input)

                l0_b_idx, l0_h_idx, l0_w_idx = get_selected_idx(self.batch_size, self.out_H_list[0],
                                                                       self.out_W_list[0], self.stride_H_list[0],
                                                                       self.stride_W_list[0])
                input_points_f1 = torch.zeros_like(input_xyz_aug_proj_f1).cuda()
                l0_xyz_proj_f1 = input_xyz_aug_proj_f1[l0_b_idx.cuda().long(),l0_h_idx.cuda().long(), l0_w_idx.cuda().long(), :]
        except Exception as e:
            rospy.logerr(f"pc processing failed: {str(e)}")



        t2=time.time()
        #new_points, new_xyz = self.model(input_xyz_aug_proj_f1, input_points_f1, l0_xyz_proj_f1)
        #new_points = self.model(input_xyz_aug_proj_f1, input_points_f1, l0_xyz_proj_f1)
     


        #print("测试成功!")
        #print(f"输入形状: input_xyz_aug_proj_f1={input_xyz_aug_proj_f1.shape}, input_points_f1={input_points_f1.shape},l0_xyz_proj_f1={l0_xyz_proj_f1.shape}")
        #print(f"输出形状: new_points={new_points.shape}")
        
        """
        
        输入形状: input_xyz_aug_proj_f1=torch.Size([1, 64, 1800, 3]), input_points_f1=torch.Size([1, 64, 1800, 3]),l0_xyz_proj_f1=torch.Size([1, 16, 225, 3])
        输出形状: new_points=torch.Size([1, 16])

        
        """
   

        #"""

        input_xyz_aug_proj_f1=input_xyz_aug_proj_f1[:,:,600:1200,:]
        mask_xyz_f1=mask_xyz_f1[:,:,600:1200,:]
        im_numpy = input_xyz_aug_proj_f1.squeeze(0).cpu().numpy()
        im_mask = mask_xyz_f1.squeeze(0).cpu().numpy()
        #print(f"im_numpy形状={im_numpy.shape}, im_mask={im_mask.shape}")
        #visualize_pc_project(im_numpy, im_mask, title="Example Cylindrical Projection")
        #"""

        self.input_xyz_aug_proj_f1=input_xyz_aug_proj_f1
        self.input_points_f1=input_points_f1
        self.l0_xyz_proj_f1=l0_xyz_proj_f1
        #print("input_xyz_aug_proj_f1"+str(self.input_xyz_aug_proj_f1.shape))
 
        



        t1=time.time()
        #print(t1-t2)
        
        # Here you would process the predictions and publish the segmented point cloud
        # For simplicity, we just publish the original point cloud
        #self.segmented_cloud_pub.publish(predictions)
    
        if self.input_xyz_aug_proj_f1 == None:
            print("input_xyz_aug_proj_f1_NONE")
        else:
            print("input_xyz_aug_proj_f1"+str(self.input_xyz_aug_proj_f1.shape))


        print("context_queue"+str(len(self.context_queue)))
        
        self.solve()

    def image_callback(self, msg):
        """ROS图像回调函数"""
        
        try:
            # 转换ROS图像为PILImage格式
            cv_image =self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)) 
            #print(pil_image.size)

            self.context_queue.append(pil_image)

            while len(self.context_queue) < self.context_size + 2:
                self.context_queue.append(pil_image)
            

            while len(self.context_queue) >= self.context_size + 2:
                    self.context_queue.pop(0)

            #print("image length"+str(len(self.context_queue)))   
        except Exception as e:
            rospy.logerr(f"Image processing failed: {str(e)}")



    def semantic_callback(self, msg):
        """ROS图像回调函数"""
        
        try:
            # 转换ROS图像为PILImage格式
            cv_image =self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.semantic_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)) 
 
        except Exception as e:
            rospy.logerr(f"Image processing failed: {str(e)}")

        


    def goal_callback(self, msg):
        """处理目标路径的回调函数"""
        try:
            # 提取第一个位姿的坐标
            first_pose = msg.poses[0].pose.position
            goal_coords = torch.tensor(
                [[first_pose.x, first_pose.y]],
                dtype=torch.float32,
                device=self.device
            )
            self.current_goal = goal_coords
        
            # Calculate distance 
            # Get last point from naction (shape: [num_samples, waypoints, 2])
            last_points = self.naction[:, -1, :]  # shape: [num_samples, 2]
            
            # Calculate goal_cost between each last point and current goal
            goal_cost = torch.norm(last_points - self.current_goal, dim=1)  # shape: [num_samples]
            
            
        except Exception as e:
            rospy.logerr(f"Error processing goal: {str(e)}")
            self.current_goal = None


    def get_action(self, diffusion_output):
        # diffusion_output: (B, 2*T+1, 1)
        # return: (B, T-1)
        action_stats=self.ACTION_STATS
        device = diffusion_output.device
        ndeltas = diffusion_output
        ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
        ndeltas = to_numpy(ndeltas)
        ndeltas = unnormalize_data(ndeltas, action_stats)
        actions = np.cumsum(ndeltas, axis=1)
        return from_numpy(actions).to(device)



    def solve(self):

        if self.input_xyz_aug_proj_f1 != None and self.input_points_f1 != None and self.l0_xyz_proj_f1 != None and len(self.context_queue)!= 0 :



            print("start solve")
            num_samples=8
            waypoint=8
    


            obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
            obs_images = obs_images.to(self.device)
            mask = torch.ones(1).long().to(self.device) # ignore the goal

            # infer action
            with torch.no_grad():
                    # encoder vision features
                start_time = time.time()
                print(obs_images.shape)
                
                obs_cond = self.model('vision_encoder', obs_img=obs_images, input_goal_mask=mask)           
                point_feature = self.model("pc_net", xyz_proj=self.input_xyz_aug_proj_f1, points_proj=self.input_points_f1, xyz_sampled_proj=self.l0_xyz_proj_f1)
                nagent_poses=torch.rand(1, 2).to(self.device, dtype=torch.float32)
                obs_cond = torch.cat([obs_cond, point_feature, nagent_poses], dim=-1)
                obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)




                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
                naction = noisy_action

                num_diffusion_iters=10
                # init scheduler
                self.noise_scheduler.set_timesteps(num_diffusion_iters)

                
                for k in self.noise_scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = self.model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                        # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                            sample=naction
                            ).prev_sample
                print("time elapsed:", time.time() - start_time)

            naction = to_numpy(self.get_action(naction))
            #print(naction)
            print(naction.shape)
            #naction = naction[0] # change this based on heuristic
            #chosen_waypoint = naction[waypoint]

            #if self.model_params["normalize"]:
            #    chosen_waypoint *= (MAX_V / RATE)
            #print(chosen_waypoint)
            naction*= (self.MAX_V / self.RATE)
            print("naction"+str(naction.shape))
            self.naction=naction
            #self.calculate_semantic_cost(naction[0])


    def calculate_semantic_cost(self, naction):
        image_ga = self.semantic_image



        x_t = []
        y_t = []
        # height to the camera frame from the ground level
        height = 0.2
        y_t = naction[:, 0]
        x_t = naction[:, 1]
        x_t = -x_t

        h_vec = np.ones(len(x_t)) * height
        points = np.transpose([x_t, h_vec, y_t])
        #print(points)

        X0 = np.ones((points.shape[0], 1))
        pointsnew = np.hstack((points, X0))
    
    
        # Camera intrinsics
        P = [[267.6525, 0.0, 337.66, 0.0], [0.0, 267.575, 186.75025, 0.0], [0.0, 0.0, 1.0, 0.0]]  # left_vga
       

        uvw = np.dot(P, np.transpose(pointsnew))
        u_vec = uvw[0]
        v_vec = uvw[1]
        w_vec = uvw[2]
        x_vec = u_vec / w_vec
        y_vec = v_vec / w_vec
        imagepoints = np.array(self.merge(x_vec, y_vec))
        # print(imagepoints)
        # imag = pred_img


        # Drawing trajectory lines through the obtained 2Dpoints in the image plane
        """
        imag = cv2.polylines(image_ga,
                         [imagepoints],
                         isClosed=False,
                         color=(0, 255, 0),
                         thickness=3,
                         lineType=cv2.LINE_AA)
        """

        # calculate semantic cost
        total_cost = 0
        for point in imagepoints:
            x, y = point
            if 0 <= y < image_ga.shape[0] and 0 <= x < image_ga.shape[1]:
                pixel = image_ga[int(y), int(x)]
                b, g, r = pixel
                if (r, g, b) == (0, 255, 0):
                    total_cost += 4
                elif (r, g, b) == (128, 128, 128):
                    total_cost += 50
        #print("Total cost:", total_cost)
        return total_cost


    def merge(x, y):
        return np.int32([x, y]).T

if __name__ == '__main__':
    rospy.init_node('pointnet_node', anonymous=True)
    pn_ros = PointNetROS()
    rospy.spin()
