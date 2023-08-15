

import math
from math import pi, sqrt, pow, exp
import time
import sys
import os
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from rclpy.clock import Clock
import torch.nn.functional as F
from torch.distributions import Normal
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
from rclpy.action import GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from custom_interfaces2.action import MaplessNavigator
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from std_msgs.msg import Float64

class PolicyDSAC(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, DEVICE):
        super(PolicyDSAC, self).__init__()
        # if agent == 'DSAC':
        self.fc_1 = nn.Linear(state_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.fc_2 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)
        self.fc_3 = nn.Linear(512, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)

        self.lr = actor_lr
        self.dev = DEVICE

        
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.to(self.dev)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    def forward(self, x):
        # if agent ==1: # DSAC 
        x= F.relu(self.norm1(self.fc_1(x)))
        x= F.relu(self.norm2(self.fc_2(x)))
        x = F.leaky_relu(self.fc_3(x))       
        mu      = self.fc_mu(x)
        log_std = self.fc_std(x)        
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std
    
    def sample(self, state, zone_box, name_agent):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        if name_agent =='SAC':
            reparameter = Normal(-mean, std)
        else:
            reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        # y_t = torch.tanh(mean)
        y_t = torch.tanh(x_t)
        if zone_box:
            # print("decrease velocitys")
            self.max_action = torch.FloatTensor([0.5, 0.3]).to(self.dev)
            self.min_action = torch.FloatTensor([-0.5, 0]).to(self.dev)
        else: 
            self.max_action = torch.FloatTensor([0.5, 0.4]).to(self.dev)
            self.min_action = torch.FloatTensor([-0.5, 0.0]).to(self.dev)
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias  = (self.max_action + self.min_action) / 2.0
        action = self.action_scale * y_t + self.action_bias
        # print("action:" + str(action))
            
        # action = torch.clamp(action[1], 0.0, 0.8)
        # print("action:" + str(action))
        return action, y_t

class PolicySAC(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, DEVICE):
        super(PolicySAC, self).__init__()

        # else:
        self.fc_1 = nn.Linear(state_dim, 512)
        # self.norm1 = nn.LayerNorm(512)
        self.fc_2 = nn.Linear(512, 512)
        # self.norm2 = nn.LayerNorm(512)
        self.fc_3 = nn.Linear(512, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)
        self.lr = actor_lr
        self.dev = DEVICE

        
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        
        self.to(self.dev)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    def forward(self, x):
        # if agent ==1: # DSAC 
        x= F.relu(self.fc_1(x))
        x= F.relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))       
        mu      = self.fc_mu(x)
        log_std = self.fc_std(x)        
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std
    
    def sample(self, state, zone_box, agent):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        reparameter = Normal(-mean, std)

        x_t = reparameter.rsample()
        # y_t = torch.tanh(mean)
        y_t = torch.tanh(x_t)
        if zone_box:
            # print("decrease velocitys")
            self.max_action = torch.FloatTensor([0.5, 0.4]).to(self.dev)
            self.min_action = torch.FloatTensor([-0.5, 0]).to(self.dev)
        else: 
            self.max_action = torch.FloatTensor([0.5, 0.4]).to(self.dev)
            self.min_action = torch.FloatTensor([-0.5, 0.0]).to(self.dev)
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias  = (self.max_action + self.min_action) / 2.0
        action = self.action_scale * y_t + self.action_bias
        # print("action:" + str(action))
            
        # action = torch.clamp(action[1], 0.0, 0.8)
        # print("action:" + str(action))
        return action, y_t
    

class SAC_Agent:
    def __init__(self):    
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.pi = PolicyNet_1225(25, 2, 0.0001, self.DEVICE)
        self.name_agent = 'SAC'
        self.SAC_pi = PolicySAC(108, 2, 0.0001, self.DEVICE )
        self.SAC_pi.load_state_dict(torch.load("/home/ras/nana_driver/model/sacmay_05_0.8_6_nana_pretrain_EP370.pt", map_location=self.DEVICE))

    def choose_action(self, state, zone_box):
        with torch.no_grad():
            action, log_prob = self.SAC_pi.sample(state.to(self.DEVICE), zone_box, self.name_agent)
        return action
    
class DSAC_Agent:
    def __init__(self):    
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name_agent = 'DSAC'
        self.DSAC_pi = PolicyDSAC(108, 2, 0.0001, self.DEVICE)
        # self.DSAC_pi.load_state_dict(torch.load("/home/ras/nana_driver/src/mapless_navigation_ros2/mapless_navigation_ros2/dsac/model/sac0604_dsac_dis_static_env01_stochastic_initial_EP920.pt", map_location=self.DEVICE))
        # self.DSAC_pi.load_state_dict(torch.load("/home/ras/nana_driver/src/mapless_navigation_ros2/mapless_navigation_ros2/dsac/model/DSAC_0603_neutral_iqn_nana_robot_EP1335.pt", map_location=self.DEVICE))
        self.DSAC_pi.load_state_dict(torch.load("/home/ras/nana_driver/model/sacDSAC_nana_0714_neutral_iqn_108_input_tau_EP1585.pt", map_location=self.DEVICE))

    def choose_action(self, state, zone_box):
        with torch.no_grad():
            action, log_prob = self.DSAC_pi.sample(state.to(self.DEVICE), zone_box, self.name_agent)
        return action  



class MaplessNavigatorServer(Node):

    def __init__(self):
        super().__init__('MaplessNavigator_server')

        self.group = ReentrantCallbackGroup()

        self.goal = MaplessNavigator.Goal()
        qos = QoSProfile(depth=10)
        #initialize policy 
        self.SAC_agent = SAC_Agent() 
        self.DSAC_agent = DSAC_Agent() 

        # initialize subcriber 
        # self.sub_odom = self.create_subscription(Odometry,'odom', self.getOdometry, 10, callback_group=self.group)
        self.sub_odom = self.create_subscription(Odometry,'/baselink2map', self.getOdometry, 10, callback_group=self.group)
        self.scan_sub = self.create_subscription(LaserScan,'/scan/front', self.getScanData, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT), callback_group=self.group)
        # self.sub_odom = self.create_subscription(Odometry,'odom', self.getOdometry, 10)
        # # self.sub_odom = self.create_subscription(Odometry,'/baselink2map', self.getOdometry, 10, callback_group=self.group)
        # self.scan_sub = self.create_subscription(LaserScan,'/scan/front', self.getScanData, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        # Initialise publishers
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', qos)
        self._marker_publisher = self.create_publisher(Marker,'goal_visualization',10)
        # Initialise servers
        self._action_server = ActionServer(
            self,
            MaplessNavigator,
            'MaplessNavigator',
            execute_callback=self.execute_callback,
            callback_group=self.group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        self.get_logger().info("MaplessNavigator action server has been initialised.")

        self.scan = LaserScan()
        self.position = Point()
        self._current_position = Point()
        self._current_position.x = 0.0
        self._current_position.y = 0.0
        self._current_position.z = 0.0
        self.get_goalbox = False
        self.goal_index = 0
        self.agent = 1       # DSAC=1, SAC=2
        self.time_step = 0   
        self.prev_x = 0
        self.prev_y = 0
        self.heading = 0
        self.goal_x, self.goal_y = 0.0, 0.0  # goal_def()
        self.total_trial = 0
        self.collision, self.arrival = 0, 0
        self.test_eps = 2  
        self.done = False       
        self.position.y = 0.0
        self.position.x = 0.0
        self.current_distance = 0.0
        self._moved_distance = Float64()
        self._moved_distance.data = 0.0
    def getScanData(self,data):
        self.scan = data
        # print("laser scan")

    def getGoalDistace(self):
        self.position.y = round(self.position.y,2)
        self.position.x = round(self.position.x,2)
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 5)
        #goal_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        return goal_distance

    def euler_from_quaternion(self,quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    

    def getOdometry(self, odom):
        # print("get Odom")
        # self.get_logger().debug("Odom CallBack")
        self.odom = odom
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        
        # (posx, posy, posz) = (self.position.x, self.position.y, self.position.z)
        (qx, qy, qz, qw) = (orientation.x, orientation.y, orientation.z, orientation.w)    
        # print("position:",posx, posy, posz )    
        # print("orientation",qx, qy, qz, qw )
        orientation_list = [qx, qy, qz, qw]
        # print("orientation", orientation_list)
        # orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = self.euler_from_quaternion(orientation_list)
        
        self.position.y = round(self.position.y,2)
        self.position.x = round(self.position.x,2)
        # print(" odom", self.position.y , self.position.x)
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = heading
        # return self.position

        ######calculate traveled distance ########
        NewPosition = self.position
        #calculate the new distance moved, and add it to _mved_distance and 
        self._moved_distance.data += self.calculate_distance(NewPosition, self._current_position)
        # Update the current position of the robot so we have a new reference point
        # (The robot has moved and so we need a new reference for calculations)
        self.updatecurrent_position(NewPosition)        

    def updatecurrent_position(self, new_position):
        """Update the current position of the robot."""
        self._current_position.x = new_position.x
        self._current_position.y = new_position.y
        self._current_position.z = new_position.z

    def calculate_distance(self, new_position, old_position):
        """Calculate the distance between two Points (positions)."""
        x2 = new_position.x
        x1 = old_position.x
        y2 = new_position.y
        y1 = old_position.y
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist


    def getState(self, scan, done):
        scan_range = []       
        heading = self.heading
        min_range = 0.35
        max_range = 3.0
        done = False
        self.goal = False
        arrival = False
        zone_goal = False

        target_size = 0.2
        # print("scan ", len(scan.ranges))
        new_ranges=105
        # mod = np.round(len(scan.ranges)/new_ranges)-1
        # mod = 2

        filter_lidar = [] 
        for i,lidar in enumerate(scan.ranges):
            if lidar==0.0:
                pass 
            else:
                filter_lidar.append(lidar)
        mod = np.round(len(filter_lidar)/new_ranges)

        for i, item in enumerate(filter_lidar):
            #print("i, item: {} {}".format(i,item))
            if (i%mod==0 and i!= 104 and i!= 110 and i!= 100 and i!= 80  and i!=60 and i!= 66 and i!= 120 and i!=130 and i!= 140 and i!=150):  #item !=0 and 
                if item==0 or item>= max_range or item == float('Inf') or np.isinf(item):
                    scan_range.append(max_range-min_range)
                else:
                    scan_range.append(item-min_range)
                if  min_range > item and item!=0.0:
                    # done = True
                    # print(done)
                    # print("detect obstacle surrounding :")
                    self.move_base(-0.15, 0.0)
                    time.sleep(0.02)
                if len(scan_range)==105:
                    break


        a = 105 - len(scan_range)
        print("a", a)
        for i in range(0,a):
            if len(scan_range) > 0:
                scan_range.append(scan_range[-1])
            else:
                scan_range.append(min_range)
        # print("scan", len(scan_range))    



        obstacle_min_range = round(min(scan_range), 2) + min_range


                        
        # if obstacle_min_range < min_range: 
        #     done = True
            # print("obstale")
            # self.move_base(-0.1, 0.0)
        current_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        #print("obstacle_min_range", obstacle_min_range)
        #print("obstacle_angle", obstacle_angle)
        #print("current: %0.2f, %0.2f" %(self.position.x, self.position.y))
        
        thread_hold = min(scan_range[39:69]) + min_range

        # elif obstacle_min_range >= 2.0:
        if current_distance < 5.0:
            self.agent = 1
        if thread_hold <= 1.9 or min(scan_range[0:39])<=1 or min(scan_range[70:105])<=1:
            self.agent = 1  # DSAC 	
            if current_distance <= 1.0:#current_distance >= 7 or :
                self.agent = 2  # SAC 	
        else: 
            self.agent = 2  # SAC 

        a = 105 + round(self.heading/0.0305) 
        print("index a", a)
        print("dis ", scan.ranges[a])
        if scan.ranges[a] < current_distance < 9.0:
            current_dis = 0.0
            self.agent = 1

        max_value1 = np.mean(scan_range[0:35])
        max_value2 = np.mean(scan_range[75:105])

        # max_value1 = min(scan_range[0:35])
        # max_value2 = min(scan_range[75:105])
        if max_value1 >= max_value2 :
            # 
            max_value = max(scan_range[0:30])
            max_index = scan_range.index(max_value)
        else: 
            max_value = max(scan_range[80:105])
            max_index = scan_range.index(max_value)			
   


        # if min(scan_range[45:65])<0.4 or min(scan_range[20:90])<=0.1:
        #     if max_value1 > max_value2 and min(scan_range[0:15]) >= min(scan_range[90:105]): #max_value == max_value1:
        #         differ_angle = -(53-max_index)*0.875 ## ben phai 
        #         self.move_base(-0.0, -0.15)
        #         heading = differ_angle
        #         print("change heading right:", differ_angle)
        #     elif max_value1 < max_value2 and min(scan_range[0:15]) <= min(scan_range[90:105]):
        #         differ_angle = (53+max_index)*0.875  ## ben trai 
        #         self.move_base(-0.0, 0.15)
        #         print("change heading left:", differ_angle)
        #         heading = differ_angle
        #     else:
        #         time.sleep(0.1)
        #         self.move_base(-0.10, 0.0)
        #     # heading = differ_angle
        # if scan.ranges[a] < current_distance and (min(scan_range[0:20]) < 0.15 or min(scan_range[85:105]) < 0.15):
        #     print("wrong direction")
        #     heading = -heading/2	


        if current_distance <= 2.0:#current_distance >= 7 or :
            self.agent = 2  # SAC 	


        #print("current_distance:",current_distance)
        if current_distance < 2.5:
            zone_goal = True
            # print("near the goal < 0.5 ")
            if current_distance < target_size:
                self.get_goalbox = True
                arrival = True
                done = True
                self.move_base(0.0, 0.0)
                # time.sleep(1.0)
                # print("real the goal")
        #print("laser scan:",scan_range)
        constant = (max_range-min_range)*np.ones(105)
        rescale = np.divide(scan_range,constant)
        laser_input = np.round(rescale, decimals=2)
        # print("current distance:" + str(current_distance))

        relationship = [round(heading,2), round(current_distance/10,2), round(obstacle_min_range/10,2)]
        #state= rescale_laser + [heading, round(current_distance/15,2), obstacle_min_range/15]
        # print("input lidar: {}".format(laser_input))
        t = 0 
        for i in laser_input:
            if i != 10.0 or i >= 0.02:
                laser_input[t] = laser_input[t] - 0.015
            t +=1
        # print ("laser input of neural:{}".format(laser_input) )
        # laser_input = np.round(rescale, decimals=2)
        state = np.concatenate((laser_input, relationship), axis=0).tolist()
        #state = [heading, current_distance, obstacle_min_range, obstacle_angle]
        # print(" state:{}".format(state))
        # print("episode State", done)
        return state, done, arrival, zone_goal

    def show_marker_in_rviz(self, point):     #Ve duong di cua robot

        marker_ = Marker()
        marker_.header.frame_id = "odom"
        # print("frame id goal ",marker_.header.frame_id  )
        time_stamp = Clock().now()
        marker_.header.stamp = time_stamp.to_msg()
        # # # marker_.header.stamp = rospy.Time.now()
        marker_.type = marker_.ARROW
        # marker_.action = marker_.ADD


        marker_.pose.position.x = point.x
        marker_.pose.position.y = point.y
        marker_.pose.position.z = point.z
        marker_.pose.orientation.x = 0.0
        marker_.pose.orientation.y = -0.8
        marker_.pose.orientation.z = 0.0
        marker_.pose.orientation.w = 1.0


        # marker_.lifetime = rospy.Duration.from_sec(lifetime_)
        marker_.scale.x = 0.8
        marker_.scale.y = 0.2
        marker_.scale.z = 0.2
        marker_.color.a = 0.8
        red_, green_, blue_ = 0.0,0.0,0.0
        marker_.color.r = 1.0
        marker_.color.g = 0.0
        marker_.color.b = 1.0

        self._marker_publisher.publish(marker_)    

    def setReward(self, state, done, action):
        obstacle_min_range = state[107]*10
        current_distance = state[106]*10
        heading = state[105]    
        path_lenght = 0.0  
        distance_2_points = math.hypot(self.prev_x - self.position.x, self.prev_y - self.position.y) 
        if not done: 
            ## adjust## 
            distance_difference = current_distance - self.previous_distance_from_desination
            if distance_difference <= 0.0:
                # rospy.logwarn("DECREASE IN DISTANCE GOOD")
                distance_difference *= 2.0
            else:
                # rospy.logerr("INCREASE IN DISTANCE BAD")
                distance_difference *= 1.5    

            rotations_cos_sum = math.cos(heading)
            reward = -2.0 * distance_difference + (math.pi - math.fabs(heading)) / math.pi + 3*rotations_cos_sum
            if obstacle_min_range < 0.45:
                reward += -5

            distance_2_points = math.hypot(self.prev_x - self.position.x, self.prev_y - self.position.y) 
            path_lenght +=distance_2_points
        else:
            if self.get_goalbox:
                self.get_logger().info('Goal!!')
                reward = 500
                self._episode_done= True

                self.goal_distance = self.getGoalDistace()
                self.get_goalbox = False
               

                self.time_step = 0
                #time.sleep(0.2)
            else: 
                # rospy.loginfo("Collision!")
                reward = -500
                self.time_step = 0
                self.pub_cmd_vel.publish(Twist())
                self.move_base(0.0, 0.0)
                self._episode_done= True

        self.previous_distance_from_desination = current_distance
        self.previous_rotation_to_goal_diff = self.heading
        self.prev_x = self.position.x
        self.prev_y = self.position.y

        return reward

    def step(self, action, done):
        
        self.time_step += 1
        linear_speed = action[1]
        angular_speed = action[0]
        # print("linear and angular velocity", linear_speed, angular_speed)
        self.move_base(linear_speed, angular_speed)
        #print("EP:", ep, " Step:", t, " Goal_x:",self.goal_x, "  Goal_y:",self.goal_y)
        # print("position", self.position.x, self.position.y)
        self.position.y = round(self.position.y,2)
        self.position.x = round(self.position.x,2)
        state, done, arrival, zone = self.getState(self.scan, done)
        reward = self.setReward(state, done, action)
        # print("step")
        self.show_marker_in_rviz(self.desired_point) 
        return np.asarray(state), reward, done, arrival , zone, self.agent

    def goal_def(self, goal_index):


        # location = randint(0, 1)
        location = goal_index
        # print("goal_index:{}".format(self.goal_index))

        if location == 0: 
            # x,y = -7.87, 5.62
            #x,y = 5.6,-8.8 # constructive
            x,y = 7.998,5.654  # A point
            # x, y =  7.953,5.723 # A point old
        elif location == 1:
            x,y = 8.243,11.054 # B point
            # x,y = 7.962,11.073  # B point  old
            # x,y = -7.87, 5.62
            # x,y = 2.5,0.4 # constructive
            # x,y = 5.0,3.0 
        elif location == 2:
            # x,y=16.917,11.001 # C point 
            x,y=17.211,10.694 # C point 
        #     x,y = -7.87, 5.62
        #     x,y = -8.79,-3.58
        # elif location == 3:
            # x,y = -0.017,-2.932
            # x,y = 16.829,5.633  # middle C point  
        # elif location == 3:
        #     x,y = 16.286,-1.986 # CEO piont
        elif location == 3:
            # x,y = 7.953,5.723 # A point old
            x,y = 7.998,5.654  # A point
        else:
        #     # x,y = -0.017,-2.932
            x,y = 0.0,0.0 # initial point
            # x,y = 5.0,10.0
        return x, y 
    def reset(self, done):
    
        self.time_step = 0
        self._episode_done = False

        self.goal_x, self.goal_y = self.goal_def(self.goal_index) #4.5, 4.5 # goal_def()
        # self.goal_x, self.goal_y = 1.33,6.6
        # self.goal_x, self.goal_y = 5.0,13.0
        # time.sleep(0.1)
        #self.goal_x, self.goal_y= 2.0, 0.0
        # print("NEXT GOAL : ", self.goal_x, self.goal_y )
        self.desired_point = Point()
        self.desired_point.x = (self.goal_x)
        self.desired_point.y = (self.goal_y) 
        self.desired_point.z = 0.0 
        self.show_marker_in_rviz(self.desired_point)  
        # print("position", self.position.y , self.position.x)
       
        self.goal_distance = self.getGoalDistace()
        state, done, _,_ = self.getState(self.scan, done)
        self.previous_distance_from_desination = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        self.previous_rotation_to_goal_diff = self.heading
        return np.asarray(state)

    def move_base(self, linear_speed, angular_speed):
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = float(linear_speed)
        cmd_vel_value.angular.z = float(angular_speed)
        self.pub_cmd_vel.publish(cmd_vel_value)
        time.sleep(0.05)
    def rotate(self):
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = 0.0
        cmd_vel_value.angular.z = 0.0
        self.pub_cmd_vel.publish(cmd_vel_value)
        time.sleep(0.05)
        # self.get_logger().info("Ex2 Rotating for "+str(self._seconds_sleeping)+" seconds")
        # for i in range(self._seconds_sleeping):
        #     self.get_logger().info("Ex2 SLEEPING=="+str(i)+" seconds")
        #     time.sleep(1)pose

    def run(self,done,total_trial,arrival, collision):

        done = False
        zone_box = False
        s = self.reset(done)
        s = torch.FloatTensor(s)
        # if self.agent ==1:
        #     print("DSAC")
        #     real_action = self.DSAC_agent.choose_action(s,zone_box)
        # else:
        #     print("SAC")
        #     real_action = self.SAC_agent.choose_action(s,zone_box)
        real_action = self.SAC_agent.choose_action(s,zone_box)
        s_prime, r, done, arrv, zone_box, self.agent = self.step(real_action, done)                       
        # time.sleep(0.1)            
        s = s_prime
        # print("after 1 step",done)
        if done:
            if arrv:
                self.goal_index = self.goal_index + 1
                self.arrival     += 1
                s = self.reset(done) 
                time.sleep(0.5)

            else:
                self.collision   += 1
                # print("Current Trial: ",total_trial)         
            self.total_trial+=1 
            print("Current Trial: ",self.total_trial)
            # print("avoidance: ",self.collision)
            # self.done = done


    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        # Accepts or rejects a client request to begin an action
        self.get_logger().info('Received goal request :)')
        self.goal = goal_request
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accepts or rejects a client request to cancel an action
        self.get_logger().info('Received cancel request :(')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')


        CMD_START_NAV  = self.goal.CMD_START_NAV
        feedback_msg = MaplessNavigator.Feedback()
        goal_id = 1
        print("start",CMD_START_NAV)
        last_time = self.get_clock().now()

        # Start executing an action
        while( self.arrival <=6):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return MaplessNavigator.Result()

            curr_time = self.get_clock().now()
            duration = Duration()
            duration = (curr_time - last_time).nanoseconds / 1e9  # unit: s

            # feedback_msg.time_left = total_driving_time - duration
            feedback_msg.navigation_time = duration
            feedback_msg.current_distance = self._moved_distance.data
            # print(self.position)
            feedback_msg.x = self.position.x 
            feedback_msg.y = self.position.y 
            # self.get_logger().info(' time {0}'.format(feedback_msg.navigation_time))
            goal_handle.publish_feedback(feedback_msg)

            # Give vel_cmd to robot
            self.run(self.done, self.total_trial, self.arrival, self.collision)
            # time.sleep(0.15)  # unit: s

        # When the action is completed
        twist = Twist()
        self.pub_cmd_vel.publish(twist)

        goal_handle.succeed()
        result = MaplessNavigator.Result()
        result.goal_success = True
        if (result.goal_success==True):
            self.get_logger().info('Returning result: SUCCESS {0}'.format(result.goal_success))
        else:
            self.get_logger().info('FAIL: {0}'.format(result.goal_success))
        return result

 

def main(args=None):
    rclpy.init(args=args)

    MaplessNavigator_action_server = MaplessNavigatorServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor(num_threads=6)

    rclpy.spin(MaplessNavigator_action_server, executor=executor)

    MaplessNavigator_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()