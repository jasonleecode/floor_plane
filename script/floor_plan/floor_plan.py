#! /usr/bin/env python3

import os
import sys
import cv2
import time
import pickle
# import torch
import numpy as np
from skimage.morphology import medial_axis
import rospy

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker

from topometricmap.topo_map import *
from topometricmap.utils import *
from floor_plan.floorplan_extraction import *
from floor_plan.utils import *

EMBEDDING_DIM = 512

# device = 'cuda' if torch.cuda.is_available() else 'cpu'    
# print(f"Initialize torch with device {device}.")

class FloorPlan(TopometricMap):
    def __init__(self, img_path=None, model_path=None, map_path=None, name=''):
        super().__init__(name)
        self.map = load_floor_plan(img_path)
        # self.interpreter = torch.load(model_path)

        self.ProcessMap(self.map)

        self.similarity = np.zeros(self.nodes.shape[0], dtype=float)
        self.belief = np.zeros(self.nodes.shape[0], dtype=float)

    def AppendNode(self, scan: LaserScan, odom, node:Node=None, to_last=True):
        node = super().AppendNode(scan, odom, node, to_last)

        local_map = self.GenerateLocalMap(node)
        embedding = self.PsudoLocalMapInterpreter(local_map)

        # Extra properties for floor plan nodes
        node.embedding = embedding
        node.similarity = 0.0

        return node


    def ProcessMap(self, map: np.ndarray):
        """
        Translate the graph extracted from the floor plan to a topometric graph
        """

        verts, edges, dist = bitmap_to_graph(map)
        # exam
        show = np.zeros(verts.shape, dtype=np.uint8)
        for p1, p2, cost in edges:
            show = cv2.line(show, (p1[1], p1[0]), (p2[1], p2[0]), 255, 1)

        # pre-generate the grid coord for the map
        node_id = -np.ones(verts.shape, dtype=int)
        verts = verts.nonzero()
        
        # assign node id
        id=0
        t1 = time.time()
        for y, x in zip(verts[0], verts[1]):
            scan = get_laserscan(dist, (y,x), resolution=0.2)

            pose = np.asarray((x, -y, 0.0, 0.0, 0.0, 0.0))/10.0
            node = self.AppendNode(scan, pose, to_last=False)
            node_id[y, x] = node.id

            id += 1
            percentage = int(100.0*id/len(verts[0]))
            print(f"\rProcess vertices: {(percentage//10)*'*'}{percentage}%", end='')

        t2 = time.time()
        print(f"Takes {(t2-t1)*1000.0} for laserscan sampling...")

        # connect every edges
        id=0
        print('')
        for p1, p2, cost in edges:

            n1 = self.nodes[node_id[p1[0], p1[1]]]
            n2 = self.nodes[node_id[p2[0], p2[1]]]
            self.AppendEdge(n1, n2)

            id += 1
            percentage = int(100.0*id/len(edges))
            print(f"\rProcess edges: {(percentage//10)*'*'}{percentage}%", end='')

        self.MarkSimilar()

    def GenerateLocalMap(self, node):
        pts = node.OptimizedLaserScanToPoints(filter=True)

        ranges = np.linalg.norm(pts, axis=1)
        centroid = np.average(pts, axis=0, weights=ranges*ranges)
        
        tf = np.identity(4, dtype=float)
        tf[:2,3] = -centroid

        return node.OptimizedCostMap(size=128, scale=15, obstacle_cost=6.0, free_space_cost=-1.0, obstacle_thickness=1, tf=tf)

    def PsudoLocalMapInterpreter(self, map: np.ndarray):
        """
        i.e, the laserscan encoder
        """

        blur = cv2.GaussianBlur(map, (11, 11), 21)
        # embedding = np.reshape(blur, -1)

        return blur

    def LocalMapInterpreter(self, map: np.ndarray):
        """
        i.e, the laserscan encoder
        """

        map_tensor = torch.tensor(map).to(device=device)
        embedding = self.interpreter(map_tensor)

        return embedding.cpu().detach().numpy()

    def UpdateSimilarity(self, msg: LaserScan, tar=None):

        target = Node(id=self.nodes.shape[0], scan=msg)
        target_map = self.GenerateLocalMap(target)
        target_embedding = self.PsudoLocalMapInterpreter(target_map)
        target_maximum = np.sum(np.square(target_embedding))

        # re-init
        self.similarity = np.zeros(self.similarity.shape, dtype=float)

        for node in (self.nodes if tar is None else tar):

            maximum = max(target_maximum, np.sum(np.square(node.embedding)))

            self.similarity[node.id] = np.sum(node.embedding*target_embedding)/maximum

        self.similarity /= np.linalg.norm(self.similarity)

    def InitSimilarity(self):

        self.similarity = np.ones(self.similarity.shape, dtype=float)
        self.similarity /= np.linalg.norm(self.similarity)

    def InitBelief(self, id):
        self.belief = np.zeros(self.nodes.shape[0], dtype=float)
        self.belief[id] = 1.0

    def UpdateBelief(self, move=0.1):
        # uncertainty = np.tan(move/10.0)
        updated_belief = np.zeros(self.nodes.shape[0], dtype=float)

        top1 = self.nodes[np.argmax(self.belief)]

        def node_func(node, param, distance):
            nonlocal updated_belief

            updated_belief[node.id] += self.belief[node.id]*(1.0-move)
            return True

        def edge_func(a: Node, b: Node, param, distance):
            nonlocal updated_belief

            diff_a = self.belief[a.id]*move/self.neighbor[a.id]
            diff_b = self.belief[b.id]*move/self.neighbor[b.id]

            updated_belief[a.id] += diff_b
            updated_belief[b.id] += diff_a

            return None

        # start traversing
        self.TraverseBF(
            node=top1,
            node_function=node_func, 
            edge_function=edge_func,
            max_distance=-1)

        updated_belief /= np.linalg.norm(updated_belief)
        self.belief = updated_belief*self.similarity
        self.belief /= np.linalg.norm(self.belief)

    def Localize(self, top_k=1):
        ind = np.argpartition(self.belief, -top_k)[-top_k:]
        mask = self.belief[ind]>0.0
        
        return self.nodes[ind][mask]
    
    def CollectNeighborhood(self, nodes, distance=2):
        """
        Collect the neighborhood of some nodes
        """

        out=[]
        cost=[]

        def cost_function(d):
            return 1.0+0.2*np.exp(-d)

        def node_func(node, param, distance):
            
            c = cost_function(distance)

            if not node in out:
                out.append(node)
                cost.append(c)
            else:
                ind = out.index(node)
                cost[ind] = min(cost[ind], c)

            return True

        def edge_func(a: Node, b: Node, param, distance):

            return None

        # start traversing
        self.TraverseBF(
            node=nodes,
            node_function=node_func, 
            edge_function=edge_func,
            max_distance=distance)
        
        return out, cost
        

    def VisuzlizeProbability(self, z_offset=0.0):
        """
        Generate the Marker Graph of the topological map based on the optimal pose of each nodes
        """

        markers = MarkerArray()

        def node_func(node, param, distance):
            nonlocal markers

            height = self.belief[node.id]*10.0
            color = [0.0, 1.0, 0.0, 1.0]
            scale = [0.1, 0.1, height]


            header,p,q = node.PrepeareMarkerConf(z_offset=z_offset)

            marker = MakeMarker(
                ns=f'{self.name}_prob',
                id=node.id,
                header=header, 
                position=p, 
                orientation=q, 
                scale=scale, 
                color=color, 
                type=Marker.CUBE)
            
            marker.pose.position.z -= height/2.0
            markers.markers.append(marker)

            return True

        def edge_func(a: Node, b: Node, param, distance):

            # vec = b.pose_opt[:2]-a.pose_opt[:2]
            # dis = np.linalg.norm(vec)
            # vec /= dis

            # d = 0.0
            # p = a.pose_opt[:2].copy()
            # while d<dis:
            #     edges.points.append(Point(x=p[0],y=p[1]))
            #     p+=vec*self.connectivity[a.id,b.id]
            #     d+=self.connectivity[a.id,b.id]
            #     edges.points.append(Point(x=p[0],y=p[1]))
            #     p+=vec*self.connectivity[a.id,b.id]
            #     d+=self.connectivity[a.id,b.id]

            return None

        # start traversing
        self.TraverseDF(
            node_function=node_func, 
            edge_function=edge_func)

        return markers


    def MarkSimilar(self):

        def node_func(node, param, distance):

            return True

        def edge_func(a: Node, b: Node, param, distance):

            maximum = max(np.sum(np.square(a.embedding)), np.sum(np.square(b.embedding)))
            score = np.sum(a.embedding*b.embedding)/maximum
            sim = max((-score+1.0)/2.0, 0.1)
            # if np.sum(a.embedding*b.embedding)>0.75:
            self.connectivity[a.id,b.id]=sim
            self.connectivity[b.id,a.id]=sim
            return None

        # start traversing
        self.TraverseDF(
            node_function=node_func, 
            edge_function=edge_func)

