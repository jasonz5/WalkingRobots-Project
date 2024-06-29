"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

from collections import deque

import rospy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String

matplotlib.use('Qt5Agg')
# Parameters
KP = 5.0  # attractive potential gain
ETA = 15.0  # repulsive potential gain
AREA_WIDTH = 3.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 3

show_animation = True


def calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy):
    """
    计算势场图
    gx,gy: 目标坐标
    ox,oy: 障碍物坐标列表
    reso: 势场图分辨率
    rr: 机器人半径
    sx,sy: 起点坐标
    """
    # 确定势场图坐标范围：
    minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0
    # 根据范围和分辨率确定格数：
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + minx  # 根据索引和分辨率确定x坐标

        for iy in range(yw):
            y = iy * reso + miny  # 根据索引和分辨率确定x坐标
            ug = calc_attractive_potential(x, y, gx, gy)  # 计算引力
            uo = calc_repulsive_potential(x, y, ox, oy, rr)  # 计算斥力
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    """
    计算引力势能：1/2*KP*d
    """
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    """
    计算斥力势能：
    如果与最近障碍物的距离dq在机器人膨胀半径rr之内：1/2*ETA*(1/dq-1/rr)**2
    否则：0.0
    """
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def oscillations_detection(previous_ids, ix, iy):
    """
    振荡检测：避免“反复横跳”
    """
    previous_ids.append((ix, iy))

    if len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH:
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False


def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    # calc potential field
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy)

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)

    if show_animation:
        draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    previous_ids = deque()

    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        # 寻找8个运动方向中势场最小的方向
        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)
        print("location: ", xp, yp)
        # detecting local minimum
        if oscillations_detection(previous_ids, ix, iy):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break

        if show_animation:
            plt.plot(ix, iy, ".r")
            # print("next position is", ix, iy)
            plt.pause(0.01)

    print("Finished")

    return rx, ry


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    sx = 0.0  # start x position [m]
    sy = 0.5  # start y position [m]
    gx = 6.0  # goal x position [m]
    gy = 6.0  # goal y position [m]
    grid_size = 0.05  # potential grid size [m]
    robot_radius = 0.5  # robot radius [m]
    # 以下障碍物坐标是我进行修改后的，用来展示人工势场法的困于局部最优的情况：
    ox = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]  # obstacle x position list [m]
    oy = [2.0, 4.0, 1.0, 3.0, 5.0, 1.0, 5.0, 3.0, 5.0, 6.0]  # obstacle y position list [m]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    rx, ry = potential_field_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

    if show_animation:
        plt.show()


def talker():
    pub = rospy.Publisher('/cassie_planner', String, queue_size = 10)
    rospy.init_node('cassie_planner', anonymous=True)
    rate = rospy.Rate(2000)
    pointer = 1
    global rx,ry
    while not rospy.is_shutdown():
        vx =  rx[pointer] 
        vy =  ry[pointer] 
        rospy.loginfo("Vx: ",vx,"  Vy: ", vy)
        pub.publish("Vx: ",vx,"  Vy: ", vy)
        rate.sleep()

if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
    talker()
