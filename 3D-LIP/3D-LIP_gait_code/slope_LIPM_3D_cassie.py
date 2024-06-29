import numpy as np
from LIPM_3D import LIPM3D

class CassieLipm():
    def __init__(self):
        self.COM_pos_x = list()
        self.COM_pos_y = list()
        self.COM_pos_z = list()
        self.left_foot_pos_x = list()
        self.left_foot_pos_y = list()
        self.left_foot_pos_z = list()
        self.right_foot_pos_x = list()
        self.right_foot_pos_y = list()
        self.right_foot_pos_z = list()
        

    def run(self):
        COM_pos_x = list()
        COM_pos_y = list()
        COM_pos_z = list()
        left_foot_pos_x = list()
        left_foot_pos_y = list()
        left_foot_pos_z = list()
        right_foot_pos_x = list()
        right_foot_pos_y = list()
        right_foot_pos_z = list()

        #some parameters
        # Distance between pelvis and stance foot ∈ (0.5, 1) m
        # Pelvis height, pz ≥ 0.80 m
        # step witdh (0.14,0.35)
        # Distance between feet > 0.2 m
        # Swing foot pitch 0 rad
        # Mid-step swing foot clearance > 0.15 m


        # Initialize the COM position and velocity
        COM_pos_0 = [-0.4, 0.2, 0.9]
        COM_v0 = [1.0, -0.01]

        # Initialize the foot positions
        left_foot_pos = [-0.2, 0.3, 0]
        right_foot_pos = [0.2, -0.3, 0]

        delta_t = 0.02

        s_x = 0.48
        s_y = 0.3
        a = 1.0
        b = 1.0
        theta = 0.0

        height = 0.036
        slope = height/s_x

        LIPM_model = LIPM3D(dt=delta_t, T_sup=0.5)
        LIPM_model.initializeModel(COM_pos_0, left_foot_pos, right_foot_pos)

        LIPM_model.support_leg = 'left_leg' # set the support leg to right leg in next step
        if LIPM_model.support_leg == 'left_leg':
            support_foot_pos = LIPM_model.left_foot_pos
            LIPM_model.p_x = LIPM_model.left_foot_pos[0]
            LIPM_model.p_y = LIPM_model.left_foot_pos[1]
        else:
            support_foot_pos = LIPM_model.right_foot_pos
            LIPM_model.p_x = LIPM_model.right_foot_pos[0]
            LIPM_model.p_y = LIPM_model.right_foot_pos[1]

        LIPM_model.x_0 = LIPM_model.COM_pos[0] - support_foot_pos[0]
        LIPM_model.y_0 = LIPM_model.COM_pos[1] - support_foot_pos[1]
        LIPM_model.vx_0 = COM_v0[0]
        LIPM_model.vy_0 = COM_v0[1]


        step_num = 0
        total_time = 30 # seconds
        global_time = 0

        swing_data_len = int(LIPM_model.T_sup/delta_t) #25
        swing_foot_pos = np.zeros((swing_data_len, 3))
        j = 0

        switch_index = swing_data_len

        for i in range(int(total_time/delta_t)):
            global_time += delta_t

            LIPM_model.step()

            # 这里设定swing foot的空间轨迹
            if step_num >= 1:
                if LIPM_model.support_leg == 'left_leg':
                    LIPM_model.right_foot_pos = [swing_foot_pos[j,0], swing_foot_pos[j,1], swing_foot_pos[j,2]]
                else:
                    LIPM_model.left_foot_pos = [swing_foot_pos[j,0], swing_foot_pos[j,1], swing_foot_pos[j,2]]
                j += 1

            # record data   
            # 这里给俩个足端轨迹赋值
            COM_pos_x.append(LIPM_model.x_t + support_foot_pos[0])
            COM_pos_y.append(LIPM_model.y_t + support_foot_pos[1])
            left_foot_pos_x.append(LIPM_model.left_foot_pos[0])
            left_foot_pos_y.append(LIPM_model.left_foot_pos[1])
            left_foot_pos_z.append(LIPM_model.left_foot_pos[2])
            right_foot_pos_x.append(LIPM_model.right_foot_pos[0])
            right_foot_pos_y.append(LIPM_model.right_foot_pos[1])
            right_foot_pos_z.append(LIPM_model.right_foot_pos[2])


            # switch the support leg
            if (i > 0) and (i % switch_index == 0):
                j = 0

                LIPM_model.switchSupportLeg() # switch the support leg
                step_num += 1

                # theta -= 0.04 # set zero for walking forward, set non-zero for turn left and right

                if step_num >= 20: # stop forward after 5 steps
                    s_x = 0.0

                # if step_num >= 20:
                #     s_y = 0.0

                if LIPM_model.support_leg == 'left_leg':
                    support_foot_pos = LIPM_model.left_foot_pos
                    LIPM_model.p_x = LIPM_model.left_foot_pos[0]
                    LIPM_model.p_y = LIPM_model.left_foot_pos[1]
                else:
                    support_foot_pos = LIPM_model.right_foot_pos
                    LIPM_model.p_x = LIPM_model.right_foot_pos[0]
                    LIPM_model.p_y = LIPM_model.right_foot_pos[1]

                # calculate the next foot locations, with modification, stable
                x_0, vx_0, y_0, vy_0 = LIPM_model.calculateXtVt(LIPM_model.T_sup) # calculate the xt and yt as the initial state for next step

                if LIPM_model.support_leg == 'left_leg':
                    x_0 = x_0 + LIPM_model.left_foot_pos[0] # need the absolute position for next step
                    y_0 = y_0 + LIPM_model.left_foot_pos[1] # need the absolute position for next step
                else:
                    x_0 = x_0 + LIPM_model.right_foot_pos[0] # need the absolute position for next step
                    y_0 = y_0 + LIPM_model.right_foot_pos[1] # need the absolute position for next step

                LIPM_model.calculateFootLocationForNextStep(s_x, s_y, a, b, theta, x_0, vx_0, y_0, vy_0)
                # print('p_star=', LIPM_model.p_x_star, LIPM_model.p_y_star)

                # calculate the foot positions for swing phase
                if LIPM_model.support_leg == 'left_leg':
                    LIPM_model.left_foot_pos[2] = (step_num-1)*height
                else:
                    LIPM_model.right_foot_pos[2] = (step_num-1)*height

                if LIPM_model.support_leg == 'left_leg':
                    right_foot_target_pos = [LIPM_model.p_x_star, LIPM_model.p_y_star, step_num*height]
                    swing_foot_pos[:,0] = np.linspace(LIPM_model.right_foot_pos[0], right_foot_target_pos[0], swing_data_len)
                    swing_foot_pos[:,1] = np.linspace(LIPM_model.right_foot_pos[1], right_foot_target_pos[1], swing_data_len)
                    swing_foot_pos[1:swing_data_len-1, 2] = (step_num+0.1)*height
                    swing_foot_pos[1:10, 2] = np.linspace((step_num-1)*height, (step_num+0.1)*height, 9)
                    swing_foot_pos[swing_data_len-10:swing_data_len-1, 2] = np.linspace((step_num+0.1)*height,step_num*height, 9)
                else:
                    left_foot_target_pos = [LIPM_model.p_x_star, LIPM_model.p_y_star, step_num*height]
                    swing_foot_pos[:,0] = np.linspace(LIPM_model.left_foot_pos[0], left_foot_target_pos[0], swing_data_len)
                    swing_foot_pos[:,1] = np.linspace(LIPM_model.left_foot_pos[1], left_foot_target_pos[1], swing_data_len)
                    # swing_foot_pos_z  曲线拟合初始点为：(step_num-1)*height  step_num*height
                    swing_foot_pos[1:swing_data_len-1, 2] = (step_num+0.1)*height
                    swing_foot_pos[1:10, 2] = np.linspace((step_num-1)*height, (step_num+0.1)*height, 9)
                    swing_foot_pos[swing_data_len-10:swing_data_len-1, 2] = np.linspace((step_num+0.1)*height,step_num*height, 9)

        self.COM_pos_x = COM_pos_x
        self.COM_pos_y = COM_pos_y
        self.left_foot_pos_x = left_foot_pos_x
        self.left_foot_pos_y = left_foot_pos_y
        self.left_foot_pos_z = left_foot_pos_z
        self.right_foot_pos_x = right_foot_pos_x
        self.right_foot_pos_y = right_foot_pos_y
        self.right_foot_pos_z = right_foot_pos_z

        for i in range(len(COM_pos_x)):
            self.COM_pos_z.append(slope*COM_pos_x[i]+COM_pos_0[2])

        for i in range(len(COM_pos_x)-1):
            if(self.left_foot_pos_z[i+1]==0 and self.left_foot_pos_z[i+1]<self.left_foot_pos_z[i]):
                left_foot_pos_z[i+1]=left_foot_pos_z[i]
            if(self.right_foot_pos_z[i+1]==0 and self.right_foot_pos_z[i+1]<self.right_foot_pos_z[i]):
                self.right_foot_pos_z[i+1]=self.right_foot_pos_z[i]

def test1():
    swing_data_len = 25
    swing_foot_pos = np.zeros((swing_data_len, 3))
    swing_foot_pos[1:swing_data_len-1, 2] = 0.20
    swing_foot_pos[1:7, 2] = np.linspace(0, 0.20, 6)
    swing_foot_pos[swing_data_len-7:swing_data_len-1, 2] = np.linspace(0.20,0, 6)
    print(swing_foot_pos)

if __name__ == '__main__':
    # test1()
    lipm = CassieLipm()
    lipm.run()
    data_len = len(lipm.COM_pos_x)
    print(data_len)
    # yaw = np.zeros(20)
    # yaw[0:10] = np.linspace(0,20,10)
    # yaw[10: ] = 20
    # print(yaw)
    # print(lipm.COM_pos_x)
    # print("-------------------------------")
    # print(lipm.COM_pos_y)
    # print("-------------------------------")
    # print(lipm.left_foot_pos_x)
    # print("-------------------------------")
    # print(lipm.left_foot_pos_y)
    # print("-------------------------------")
    print(lipm.left_foot_pos_z)
    # print("-------------------------------")
    # print(lipm.right_foot_pos_x)
    # print("-------------------------------")
    # print(lipm.right_foot_pos_y)
    # print("-------------------------------")
    # print(lipm.right_foot_pos_z)
