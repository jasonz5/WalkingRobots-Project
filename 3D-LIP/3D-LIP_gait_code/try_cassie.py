import pybullet as p
import time
import pybullet_data
import numpy as np
from sklearn import naive_bayes
import planner
from demo_LIPM_3D_cassie import CassieLipm
# from slope_LIPM_3D_cassie import CassieLipm

class Cassie(object):
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        self.ground = p.loadURDF("plane.urdf")
        # self.stair = p.loadURDF('/newstair/urdf/newstair.urdf', [2.05, 1, 0.2], [0, 0, 0, 1], useFixedBase=True)
        # p.setGravity(0, 0, -9.81)
        p.setGravity(0, 0, 0)
        self.robot = p.loadURDF("/cassie/urdf/cassie_lipm.urdf",basePosition=[0,0,0.8],baseOrientation = [0, 0, 0, 1],useFixedBase=False)

        # p.setPhysicsEngineParameter(numSolverIterations=100)
        p.changeDynamics(self.robot,-1,linearDamping=0, angularDamping=0)

        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=[0, 0, 0.6])

        self.jointIds=[]
        self.jointBase = []
        # self.jointAngles=[0,0,1.0204,-1.97,-0.084,2.06,-1.9,0,0,1.0204,-1.97,-0.084,2.06,-1.9,0]
        self.jointAngles=[-0.800,1.811,-1.811,-0.800,1.811,-1.811]
        self.activeJoint=0
        self.joingNames = []
        self.jointTypes = []
        self.get_joints_cassie()
        # p.setRealTimeSimulation(1)
        

        # from yanshee
        self.n_j = len(self.jointIds)
        self.n_base = len(self.jointBase)
        self.simu_f = 50 # Simulation frequency, Hz
        p.setTimeStep(1.0/self.simu_f)
        self.stance_idx = 0
        self.pre_foot_contact = np.array([1, 0])
        self.foot_contact = np.array([1, 0]) 
        self.q_vec = np.zeros(self.n_j)
        self.dq_vec = np.zeros(self.n_j)
        self.q_base = np.zeros(self.n_base)
        self.dq_base = np.zeros(self.n_base)

        self.COM_pos_x = list()
        self.COM_pos_y = list()
        self.COM_pos_z = list()
        self.left_foot_pos_x = list()
        self.left_foot_pos_y = list()
        self.left_foot_pos_z = list()
        self.right_foot_pos_x = list()
        self.right_foot_pos_y = list()
        self.right_foot_pos_z = list()
        self.yaw = np.zeros(1500)
        self.yaw[0:400] = np.linspace(0,1.,400)
        self.yaw[400: ] = 1.

    def initial(self):
        for j in range(len(self.jointIds)):
            p.setJointMotorControl2(self.robot, self.jointIds[j], p.POSITION_CONTROL, self.jointAngles[j], force=140.)

    def get_joints_cassie(self):
        for j in range (p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot,j,linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robot,j)
            #print(info)
            jointName = info[1]
            self.joingNames.append(jointName)
            jointType = info[2]
            self.jointTypes.append(jointType) 
            if (jointType==p.JOINT_REVOLUTE):
                self.jointIds.append(j)
                p.resetJointState(self.robot, j, self.jointAngles[self.activeJoint])
                # p.setJointMotorControl2(self.robot, j, controlMode=p.POSITION_CONTROL, force=0)
                self.activeJoint+=1
            elif(jointType==p.JOINT_PLANAR or jointType==p.JOINT_PRISMATIC):
                self.jointBase.append(j)
            # else:
            #     print(jointType)
        print("Number of All Joints:", p.getNumJoints(self.robot))
        print("Number of All Revolute Joints:", self.jointIds)   
        print("Number of Joint Base:", self.jointBase)  
        print("Number of All types Joints:", self.jointTypes)     
        print("Joints Names: ",self.joingNames )
        print("activeJoint: ",self.activeJoint)

    def run(self):
        for i in range(len(self.COM_pos_x)):  
            # torque_array = self.controller(i)   
            # self.q_vec, self.dq_vec, self.q_base, self.dq_base = self.step(torque_array) 

            
            self.step_pos(i)
            time.sleep(0.04)
        p.disconnect()


    def step_pos(self,t):
        self.set_motor_position_array(t)
        p.stepSimulation()
        # climd or turn
        basePosition, _  = p.getBasePositionAndOrientation(self.robot)
        # basePosition[2] = 0.4
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=basePosition)

        # forward（trash）
        # self.q_base, _, _, _  = p.getJointState(self.robot, self.jointBase[0]) 
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
        #                              cameraPitch=0, cameraTargetPosition=[self.q_base,0,0.9])

    def set_motor_position_array(self, t):
        # if(t<50):
        #     com_z = 0.8
        # else:
        #     com_z = np.float32(self.COM_pos_z[t])
        qLeft  = planner.IK(np.float32(self.COM_pos_x[t]), np.float32(self.COM_pos_z[t])-0.1, np.float32(self.left_foot_pos_x[t]),  np.float32(self.left_foot_pos_z[t]))
        qRight = planner.IK(np.float32(self.COM_pos_x[t]), np.float32(self.COM_pos_z[t])-0.1, np.float32(self.right_foot_pos_x[t]), np.float32(self.right_foot_pos_z[t]))
        q_joint = np.array([qLeft[0],qLeft[1],qLeft[2],qRight[0],qRight[1],qRight[2]])
        for j in range(len(self.jointBase)):
            ## trash
            # basePosition = [np.float32(self.COM_pos_x[t]),0,np.float32(self.COM_pos_z[t])-0.2] # forword
            # baseOrientation = p.getQuaternionFromEuler([0, 0, 0 ]) # forword
            # p.setJointMotorControl2(self.robot, self.jointBase[j], p.POSITION_CONTROL, np.float32(self.COM_pos_x[t]), force=140.)  # forword
            ##
            basePosition = [np.float32(self.COM_pos_x[t]),np.float32(self.COM_pos_y[t]),np.float32(self.COM_pos_z[t])-0.2] # forword
            baseOrientation = p.getQuaternionFromEuler([0,0,0]) # forword
            # basePosition = [np.float32(self.COM_pos_x[t]),np.float32(self.COM_pos_y[t]),np.float32(self.COM_pos_z[t])]# turn
            # baseOrientation = p.getQuaternionFromEuler([0,0,self.yaw[t]]) # turn
            # basePosition = [np.float32(self.COM_pos_x[t]),0,np.float32(self.COM_pos_z[t])-0.18] # climb
            # baseOrientation = p.getQuaternionFromEuler([0, 0, 0 ]) # climb
            p.resetBasePositionAndOrientation(self.robot, posObj=basePosition,ornObj = baseOrientation)  # turn climb
            
        for j in range(len(self.jointIds)):
            # p.resetJointState(self.robot, self.jointIds[j], q_joint[j])
            p.setJointMotorControl2(self.robot, self.jointIds[j], p.POSITION_CONTROL, q_joint[j], force=140.)
        # for i in range(len(self.jointIds)):
            #     p.setJointMotorControl2(self.robot,self.jointIds[i],p.POSITION_CONTROL,self.jointAngles[i], force=140.)	

    def copy(self,lipm):
        self.COM_pos_x = lipm.COM_pos_x
        self.COM_pos_y = lipm.COM_pos_y
        self.COM_pos_z = lipm.COM_pos_z
        self.left_foot_pos_x = lipm.left_foot_pos_x
        self.left_foot_pos_y = lipm.left_foot_pos_y
        self.left_foot_pos_z = lipm.left_foot_pos_z
        self.right_foot_pos_x = lipm.right_foot_pos_x
        self.right_foot_pos_y = lipm.right_foot_pos_y
        self.right_foot_pos_z = lipm.right_foot_pos_z

    
    def controller(self, t):
        # PD controller gains
        k = np.array([100, 100, 50, 50, 100, 50, 50])
        b = np.array([0.05, 0.05, 0.05, 0.05,  0.05, 0.05, 0.05])
        qLeft = planner.IK(np.float32(self.COM_pos_x[t]),0.8,np.float32(self.left_foot_pos_x[t]), np.float32(self.left_foot_pos_z[t]))
        qRight = planner.IK(np.float32(self.COM_pos_x[t]),0.8,np.float32(self.right_foot_pos_x[t]), np.float32(self.right_foot_pos_z[t]))
        q= np.array([self.q_base[0],self.q_vec[0],self.q_vec[1],self.q_vec[2],self.q_vec[3],self.q_vec[4],self.q_vec[5]])
        q_d = np.array([np.float32(self.COM_pos_x[t]),qLeft[0],qLeft[1],qLeft[2],qRight[0],qRight[1],qRight[2]])
        return self.joint_PD_controller(k, q, q_d)

    def joint_PD_controller(self, k, q, q_d):
        return k*(q_d-q)

    def step(self, torque_array):
        self.set_motor_torque_array(torque_array)
        p.stepSimulation()
        return self.get_joint_states()

    def set_motor_torque_array(self, torque_array = None):
        if torque_array is None:
            torque_array = np.zeros(self.n_j+self.n_base)
        for j in range(len(self.jointBase)):
            p.setJointMotorControl2(self.robot, self.jointBase[j], p.TORQUE_CONTROL, force=torque_array[j])
        for j in range(len(self.jointIds)):
            p.setJointMotorControl2(self.robot, self.jointIds[j], p.TORQUE_CONTROL, force=torque_array[j+self.n_base])

    def get_joint_states(self):
        q_vec = np.zeros(self.n_j)
        dq_vec = np.zeros(self.n_j)
        q_base = np.zeros(self.n_base)
        dq_base = np.zeros(self.n_base)
        for j in range(self.n_j):
            q_vec[j], dq_vec[j], _, _  = p.getJointState(self.robot, self.jointIds[j])
        for j in range(self.n_base):
            q_base[j], dq_base[j], _, _  = p.getJointState(self.robot, self.jointBase[j])
        return q_vec, dq_vec, q_base, dq_base

if __name__ == '__main__':
    lipm = CassieLipm()
    lipm.run()
    cassie = Cassie()
    cassie.copy(lipm)
    cassie.run()
