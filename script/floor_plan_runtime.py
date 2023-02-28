#! /usr/bin/env python3
import networkx as nx

import rospy
import time
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from topometricmap.topo_runtime import *
from topometricmap.topo_map import *
from topometricmap.utils import *
from floor_plan.floor_plan import *
from floor_plan.floorplan_extraction import *
from floor_plan.utils import *

EMBEDDING_DIM = 512

scan = None

def scan_cb(msg: LaserScan):
    global scan
    scan = msg
    scan.angle_min += euler[2]
    scan.angle_max += euler[2]

euler = np.asarray((0.0,0.0,0.0), dtype=float)
movement = 0.0
def odom_cb(msg: Odometry):
    global euler, movement
    q = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
    movement += np.linalg.norm(np.asarray( (msg.twist.twist.linear.x, msg.twist.twist.linear.y), dtype=float))
    euler = euler_from_quaternion(q, axes='sxyz')


def ShowCorrespondence(floor:FloorPlan, map:TopoMapRuntime, correspondence, z0=0.0, z1=0.0):

    markers = MarkerArray()

    edges = MakeMarker(
        ns='correspondence', 
        id=0, 
        header=Header(
            frame_id=frame_id, 
            stamp=rospy.Time.now()), 
        type=Marker.LINE_LIST, 
        scale=0.01)

    markers.markers.append(edges)

    for n in map.nodes:
        if correspondence[n.id]<0:
            continue
        c = floor.nodes[correspondence[n.id]]
        edges.points.append(Point(x=n.pose_opt[0],y=n.pose_opt[1], z=z0))
        edges.points.append(Point(x=c.pose_opt[0],y=c.pose_opt[1], z=z1))

    return markers

def node_subst_cost(n1, n2):
    target_map = floor_plan.GenerateLocalMap(n2["node"])
    target_embedding = floor_plan.PsudoLocalMapInterpreter(target_map)
    target_maximum = np.sum(np.square(target_embedding))

    maximum = max(target_maximum, np.sum(np.square(n1["node"].embedding)))

    similarity = np.sum(n1["node"].embedding*target_embedding)/maximum

    return 1.0-similarity

if __name__ == "__main__":
    rospy.init_node("floor_plan", anonymous=True)
    marker_pub = rospy.Publisher("floor_plan", MarkerArray, queue_size=10)

    map = TopoMapRuntime('topometric')

    floor_plan: FloorPlan = LoadTopomap("map.pickle") #FloorPlan("/home/rtu/catkin_clip_topo/src/floor_plane/topdown_floors3.png")
    # floor_plan: FloorPlan = FloorPlan("/home/rtu/catkin_clip_topo/src/floor_plane/topdown_floors3.png")
    # floor_plan.Save('map.pickle')
    floor_plan.name = 'floor_plan'
    floor_plan.InitBelief(35)
    floor_plan.InitSimilarity()
    floor_plan.PropogatePosition(floor_plan.nodes[39], max_distance=-1)

    rospy.Subscriber("scan_aug", LaserScan, callback=scan_cb, queue_size=1)
    rospy.Subscriber("odom", Odometry, callback=odom_cb, queue_size=1)

    rospy.Rate(1).sleep()

    markers = floor_plan.Visuzlize(show_laser=True, max_distance=-1, z_offset=-5.0)
    marker_pub.publish(markers)

    floor_plan_nx = floor_plan.AsNetworkXGraph()
    print(floor_plan_nx)

    # prev_anchor = map.anchor
    
    correspondence = np.full(1024, -1, dtype=int)
    correspondence[0] = 25
    while not rospy.is_shutdown():
        
        map.Step()

        if len(map.nodes)>3:
            map_nx = map.AsNetworkXGraph()
            print(map_nx)
            paths, cost = nx.optimal_edit_paths(floor_plan_nx, map_nx, node_subst_cost=node_subst_cost)
            print(paths)
            print(cost)

        # floor_plan.belief = np.zeros(floor_plan.nodes.shape[0], dtype=float)

        # # Maybe collect all the corresponding nodes of the neighbor?
        # # nodes = floor_plan.Localize(3)
        # # print(nodes)
        # # nodes = floor_plan.nodes[correspondence[prev_anchor.id]]
        
        # nodes = []
        # for n in map.Neighborhood(map.anchor, max_distance=4):
        #     if correspondence[n.id]<0:
        #         continue
        #     nodes.append(floor_plan.nodes[correspondence[n.id]])

        # candidates, cost = floor_plan.CollectNeighborhood(nodes, distance=5)
        # floor_plan.UpdateSimilarity(scan, candidates)

        # possibility = np.zeros(len(candidates), dtype=float)
        # for i,(n,c) in enumerate(zip(candidates, cost)):
        #     print(f"{n.id}, ", end='')
        #     possibility[i] = floor_plan.similarity[n.id]
        #     floor_plan.belief[n.id] = possibility[i]
        # print("")

        # e = np.exp(possibility)
        # possibility = e/np.sum(e)
        # best = np.argmax(possibility)
        # # if possibility[best]>0.1:
        # print(f"Best:{candidates[best].id} with {possibility[best]}")
        # correspondence[map.anchor.id] = candidates[best].id

        # connections = ShowCorrespondence(floor_plan, map, correspondence, 0.0, -5.0)
        # if len(connections.markers)>0:
        #     marker_pub.publish(connections)

        # t1 = time.time()
        # if scan is not None and movement>2.0:

        #     # align the
        #     scan.angle_min += euler[2]
        #     scan.angle_max += euler[2]
        #     # floor_plan.UpdateSimilarity(scan)
        #     # floor_plan.UpdateBelief(move=0.35)
        #     movement=0.0
        # t2 = time.time()

        markers = floor_plan.VisuzlizeProbability(z_offset=-5.0)
        marker_pub.publish(markers)
        # t3 = time.time()

        # print(f"Update time: {(t2-t1)*1000.0}")
        # print(f"Visualize time: {(t3-t2)*1000.0}")

        rospy.Rate(10).sleep()



