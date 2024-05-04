#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import struct
import numpy as np
import open3d as o3d
import scipy
import tensorflow as tf


decoder = {0: 'cyclist', 1: 'misc', 2: 'pedestrian', 3: 'vehicle'}

# fields = [PointField('x', 0, PointField.FLOAT32, 1),
#           PointField('y', 4, PointField.FLOAT32, 1),
#           PointField('z', 8, PointField.FLOAT32, 1),
#         #   PointField('rgb', 12, PointField.FLOAT32, 1),
#           PointField('rgba', 12, PointField.UINT32, 1)
#           ]

class PointNet:
    def __init__(self) -> None:
        
        # self.session = tf.Session()
        self.print = True
        try:
            self.model=tf.keras.models.load_model("/home/tarun/cop_ws/src/pointnet_ros/scripts/model/pointnet_model.h5")
            rospy.loginfo("MODEL LOADED")
        except:
            rospy.loginfo("SOME EXEPTION WAS CAUGHT WITH LOADING POINTNET MODEL")

        self.sub = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.callback,queue_size=20)
        # self.pub_ground = rospy.Publisher("pointnet_ground", PointCloud2 ,queue_size=1)
        self.class_pub=[]
        for i in decoder:
            self.class_pub.append(rospy.Publisher(f"class{i}", PointCloud2 ,queue_size=1))

    def predict(self,segmented_objects, object_ids, model, config):
        dataset, N = self.preprocess(segmented_objects, object_ids, config)

        predictions = {
            class_id: {} for class_id in range(config['num_classes'])
        }
        num_predicted = 0

        for X, y in dataset:
            prob_preds = model(X)
            ids = y.numpy()

            for (object_id, class_id, confidence) in zip(
                ids,
                np.argmax(prob_preds, axis=1),
                np.max(prob_preds, axis=1)
            ):
                predictions[class_id][object_id] = confidence
                num_predicted += 1

                if (num_predicted == N):
                    break
        
        return predictions
    
    def preprocess(
        self,segmented_objects, object_ids,
        config
        ):
        points = np.asarray(segmented_objects.points)
        normals = np.asarray(segmented_objects.normals)
        num_objects = max(object_ids) + 1

        X = []
        y = []

        for object_id in range(num_objects):
            if ((object_ids == object_id).sum() <= 4):
                continue
            
            # 2. only keep object within max radius distance:
            object_center = np.mean(points[object_ids == object_id], axis=0)[:2]
            if (np.sqrt((object_center*object_center).sum()) > config['max_radius_distance']):
                continue
            
            points_ = np.copy(points[object_ids == object_id])
            normals_ = np.copy(normals[object_ids == object_id])
            N, _ = points_.shape

            weights = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(points_, 'euclidean')
            ).mean(axis = 0)
            weights /= weights.sum()
            
            idx = np.random.choice(
                np.arange(N), 
                size = (config['num_sample_points'], ), replace=True if config['num_sample_points'] > N else False,
                p = weights
            )

            points_processed, normals_processed = points_[idx], normals_[idx]
            points_processed -= points_.mean(axis = 0)

            X.append(
                np.hstack(
                    (points_processed, normals_processed)
                )
            )
            y.append(object_id)

        X = np.asarray(X)
        y = np.asarray(y)

        N = len(y)
        if (N % config['batch_size'] != 0):
            num_repeat = config['batch_size'] - N % config['batch_size']

            X = np.vstack(
                (
                    X, 
                    np.repeat(
                        X[0], num_repeat, axis=0
                    ).reshape(
                        (-1, config['num_sample_points'], 6)
                    )
                )
            )
            y = np.hstack(
                (y, np.repeat(y[0], num_repeat))
            )

        # format as tensorflow dataset:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(X, dtype=tf.float32), 
                tf.convert_to_tensor(y, dtype=tf.int64)
            )
        )
        dataset = dataset.batch(batch_size=config['batch_size'], drop_remainder=True)

        return dataset, N
    
    def segment_ground_and_objects(self,point_cloud):
        x=rospy.get_time()
            

        N, _ = point_cloud.shape
        #print("1a:",(y:=rospy.get_time())-x)

        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_original.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=5.0, max_nn=9
            )
        )
        #print("2a:",(x:=rospy.get_time())-y)

        normals = np.asarray(pcd_original.normals)
        angular_distance_to_z = np.abs(normals[:, 2])
        idx_downsampled = angular_distance_to_z > np.cos(np.pi/6)
        #print("3a:",(y:=rospy.get_time())-x)

        pcd_downsampled = o3d.geometry.PointCloud()
        pcd_downsampled.points = o3d.utility.Vector3dVector(point_cloud[idx_downsampled])

        ground_model, idx_ground = pcd_downsampled.segment_plane(
            distance_threshold=0.30,
            ransac_n=3,
            num_iterations=100
        )
        #print("4a:",(x:=rospy.get_time())-y)


        segmented_ground = pcd_downsampled.select_by_index(idx_ground)

        distance_to_ground = np.abs(
            np.dot(point_cloud,np.asarray(ground_model[:3])) + ground_model[3]
        )
        idx_cloud = distance_to_ground > 0.30

        segmented_objects = o3d.geometry.PointCloud()
        #print("5a:",(y:=rospy.get_time())-x)

        idx_segmented_objects = np.logical_and.reduce(
            [
                idx_cloud,
                point_cloud[:, 0] >=   1.95, point_cloud[:, 0] <=  80.00,
                point_cloud[:, 1] >= -30.00, point_cloud[:, 1] <= +30.00
            ]
        )

        segmented_objects.points = o3d.utility.Vector3dVector(
            point_cloud[idx_segmented_objects]
        )
        segmented_objects.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd_original.normals)[idx_segmented_objects]
        )
        #print("6a:",(x:=rospy.get_time())-y)

        segmented_ground.paint_uniform_color([0.0, 0.0, 0.0])
        segmented_objects.paint_uniform_color([0.5, 0.5, 0.5])

        labels = np.asarray(segmented_objects.cluster_dbscan(eps=0.60, min_points=3))
        #print("7a:",(y:=rospy.get_time())-x)

        return segmented_ground, segmented_objects, labels


    def publish_result(self,segmented_objects, object_ids, predictions, decoder,pub_list):
        
        points = np.asarray(segmented_objects.points)
        # rgb = struct.unpack('I', struct.pack('BBBB', 0, 0, 255, 255))[0]
        for class_id in predictions:
            class_name = decoder[class_id]

            if (class_name != 'vehicle'):
                continue

            all_pts=[]
            for object_id in predictions[class_id]:
                
                if predictions[class_id][object_id] < 0.9:
                    continue

                # create point cloud:
                object_pcd = o3d.geometry.PointCloud()
                object_pcd.points = o3d.utility.Vector3dVector(
                    points[object_ids == object_id]
                )
                header = Header()
                header.frame_id = 'base_link'
                header.stamp = rospy.Time.now()
                a=list(object_pcd.points)
                all_pts=all_pts+a
            
            # print(rgb)
            np_pts=np.array(all_pts)
            # print(x:=np.full((np_pts.shape[0], 1), rgb))
            # coloured_pc2 = np.concatenate((np_pts,x),axis=1) 
            # print(coloured_pc2)
            # combined = pc2.create_cloud(header, fields, coloured_pc2)
            combined = pc2.create_cloud_xyz32(header, np_pts)
            # print(combined)
            header = Header()
            header.frame_id = 'base_link'
            header.stamp = rospy.Time.now()
            pub_list[class_id].publish(combined)
        
        

    
    def callback(self, ros_point_cloud):
        x=rospy.get_time()
        points = np.array(list(pc2.read_points(ros_point_cloud,field_names=("x", "y", "z"), skip_nans=True)))
        #print("1:",(y:=rospy.get_time())-x)
        segmented_ground,segmented_objects,object_ids=self.segment_ground_and_objects(points)
        #print("2:",(x:=rospy.get_time())-y)
        # header = Header()
        # header.frame_id = 'base_link'
        # header.stamp = rospy.Time.now()
        

        # combined = pc2.create_cloud_xyz32(header, segmented_ground.points)
        # self.pub_ground.publish(combined)
        config = {
        'max_radius_distance': 25.0,
        'num_sample_points': 64,
        'msg' : True,
        'batch_size' : 16,
        'num_classes' : 4
        }
        #print("3:",(y:=rospy.get_time())-x)
        predictions = self.predict(segmented_objects, object_ids, self.model, config)
        #print("4:",(x:=rospy.get_time())-y)
        #print(predictions)
        #print("5:",(y:=rospy.get_time())-x)
        self.publish_result(
            segmented_objects, object_ids, 
            predictions, decoder,self.class_pub
        )
        #print("6:",(x:=rospy.get_time())-y)

def main():
    rospy.init_node("pointnet")

    try:
        PointNet()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()


if __name__ == '__main__':
    main()