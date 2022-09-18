from os.path import isfile, join
from os import listdir
from datautils.DatasetParser import DatasetParser
from poseutils.Pose import Pose
import numpy as np
import transforms3d as t3d
import logging
from skimage.io import imread


class MegaDepthParser(DatasetParser):
    """
       A class used to represented a parser of the MegaDepthParser dataset
       Attributes
        ----------
        scenes : list <str>
            a list of scenes in the dataset
    """

    def __init__(self, name, dataset_path):
        """
        :param name: (str) the name of the parser
        :param dataset_path: (str) the path to the physical location of the dataset
        """
        super(MegaDepthParser, self).__init__(name, dataset_path)
        self.model_format = 'h5_depth'

    def parse_dataset(self):
        """
        Parse the dataset associated with the parser
        :return: a dictionary of the following structure
                scene_name -> split_name -> seq_name -> [img_paths, poses, 3d_models_paths, intrinsic_cameras, extrinsic_cameras]
        """
        dataset_dict = {}
        scene_folders = [f for f in listdir(join(self.dataset_path, 'Undistorted_SfM'))]
        for scene in scene_folders:
            logging.info("Start to parse scene {}".format(scene))
            # Load npz and keys
            scene_metadata_path = join(self.dataset_path, join('metadata', "{}.0.npz".format(scene)))
            if not isfile(scene_metadata_path):
                logging.info("scene: {} is not a valid scene (missing metada)".format(scene))
                continue
            scene_metadata = np.load(scene_metadata_path, allow_pickle=True)
            keys = scene_metadata.files

            # Set scene name and other identifiers
            split = 'train'
            seq = 'seq-all'

            # Fetch the data
            all_imgs_paths = scene_metadata[keys[0]]
            all_models_paths = scene_metadata[keys[1]]
            all_intrinsic_cameras = scene_metadata[keys[2]]
            all_extrinsic_cameras = scene_metadata[keys[3]]

            imgs_paths = []
            poses = []
            models_paths = []
            intrinsic_cameras = []
            extrinsic_cameras = []
            for i in range(len(all_imgs_paths)):
                img_path = all_imgs_paths[i]
                if img_path is None:
                    assert all_models_paths[i] is None
                    assert all_extrinsic_cameras[i] is None
                    assert all_intrinsic_cameras[i] is None
                else:
                    if isfile(join(self.dataset_path,img_path)):
                        try:
                            imread(join(self.dataset_path,img_path))
                            imgs_paths.append(all_imgs_paths[i])
                            models_paths.append(all_models_paths[i])
                            intrinsic_cameras.append(all_intrinsic_cameras[i])
                            pose = np.linalg.inv(all_extrinsic_cameras[i])
                            extrinsic_cameras.append(all_extrinsic_cameras[i])
                            t, rotm, _, _ = t3d.affines.decompose(pose)
                            poses.append(Pose(t=t, rotm_R=rotm))
                        except ValueError:
                            logging.info("Corrupted image at: {}".format(img_path))
                    else:
                        logging.info("MegaDepthParser: file: {} listed in expected files but not found".format(img_path))

            dataset_dict[scene] = {split: {seq:[imgs_paths, poses, models_paths, intrinsic_cameras, extrinsic_cameras]}}

        return dataset_dict
