import numpy as np
import warnings

# try:
#     from contact_graspnet.contact_grasp_estimator import (
#         GraspEstimator as ContactGraspEstimator,
#     )
#     import tensorflow.compat.v1 as tf

# except ImportError:
#     warnings.warn("Contact Graspnet not found in this env. Skipping import")
from contact_graspnet.contact_grasp_estimator import (
    GraspEstimator as ContactGraspEstimator,
)
import tensorflow.compat.v1 as tf
from contact_graspnet.config_utils import load_config
from addict import Dict


class ContactGraspnetPipeline:
    def __init__(self, config, is_debug=False):
        self.model_config = load_config(
            checkpoint_dir=config.model_config.checkpoint_dir,
            batch_size=config.model_config.forward_passes,
        )

        self.target_pc_size = config.target_pc_size
        self.threshold = config.threshold
        self.num_grasp_samples = config.num_grasp_samples

        self.gripper_ctrl_pts_file = config.gripper_ctrl_pts_file
        # self.choose_fn = self.config.choose_fn

        self.config = config

        self.estimator = None

        self.is_debug = is_debug

        self.mode = "object"  # [ "object", "scene"]

        # TF session
        self.sess = None

    def initialize(self):
        # # Start the grasp_pose_publisher
        # self.grasp_pose_publisher = rospy.Publisher(
        #     "grasp_pose", PoseStamped, queue_size=1, latch=True
        # )

        # self.pregrasp_pose_publisher = rospy.Publisher(
        #     "pregrasp_pose", PoseStamped, queue_size=1, latch=True
        # )

        checkpoint_dir = self.config.model_config.checkpoint_dir

        self.grasp_estimator = ContactGraspEstimator(
            cfg=self.model_config,
        )
        # self.grasp_estimator._num_input_points = self.target_pc_size
        self.model_config["TEST"]["with_replacement"] = True
        self.model_config["TEST"]["num_samples"] = self.num_grasp_samples

        self.grasp_estimator.build_network()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode="test")

        return

    def test_initialization(self):
        info_msg = (
            "----------------------------------------------------\n"
            "Grasp Pipeline (Contact Graspnet) Initialization Test.\n"
            "Passing a dummy pointcloud and checking if the right number of grasps are received.\n"
            " -------------------- BEHAVIOR ---------------------\n"
            "If debug mode is active, trimesh window will open with pointcloud and grasps visualized\n"
            " ---------------------------------------------------\n"
        )
        # rospy.loginfo(info_msg)

        pc = np.random.rand(2048, 3)
        num_grasps = 10

        grasps, scores, selected_idcs, _, _ = self.get_grasps(pc, num_grasps)

        # num_out_grasps = grasps.shape[0] * grasps.shape[1]

        # assert (
        #     num_out_grasps == num_grasps
        # ), f"Expected {num_grasps} grasps, got {num_out_grasps}"

        # out_grasps = self.transform_to_fingertip_frame(results["grasps"].clone())

        return True

    def get_grasps(self, pc: np.ndarray, num_grasps: int = 10) -> np.ndarray:
        pc_clone = pc.copy()
        pc = pc[np.random.permutation(pc.shape[-2])][: self.target_pc_size]
        grasps_cam, scores, selected_idcs, _, _ = self.grasp_estimator.predict_grasps(
            self.sess, pc, convert_cam_coords=True
        )

        # Sort by scores
        if selected_idcs.size == 0:
            warnings.warn("No grasps found on the pointcloud")
            return None, None
        else:
            grasps_cam = grasps_cam[selected_idcs]
            scores = scores[selected_idcs]

        # Select top grasps
        sorted_idcs = np.argsort(scores)[::-1]
        grasps_cam = grasps_cam[sorted_idcs]
        scores = scores[sorted_idcs]

        # if self.is_debug:
        #     self.debug_visualize(pc_clone, grasps_cam)

        return grasps_cam, scores

    def transform_to_fingertip_frame(self, grasps):
        """
        Transform grasps to fingertip frame

        Args:
            grasps (np.ndarray): (N, 4, 4) grasp poses
                        H = [R | t]
                            [0 | 1]
        """

        transform = np.eye(4)

        # Here we transform z to the fingertip according
        # to the control points defined in  {contact_graspnet_dir}/gripper_control_points/contact_graspnet/gripper_control_points/panda_gripper_coords.yml
        transform[..., :3, 3] += np.array([0, 0, 0.1034])

        grasps = grasps @ transform

        return grasps


if __name__ == "__main__":
    config = Dict(
        {
            "model_config": {
                "checkpoint_dir": "checkpoints/scene_test_2048_bs3_hor_sigma_001",
                "z_range": [0.2, 1.8],
                "filter_grasps": False,
                "forward_passes": 1,
            },
            "target_pc_size": 2048,
            "threshold": 0.5,
            "num_grasp_samples": 100,
        }
    )

    pipeline = ContactGraspnetPipeline(config)
    pipeline.initialize()
    pipeline.test_initialization()
