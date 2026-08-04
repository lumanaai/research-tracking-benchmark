import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import albumentations as alb
import cv2
from general.analyzer_general import InferenceType
from general.img_utils import ResizeAndPadToTarget
from general.inference import BaseInferenceConfig, InferenceWrapper
from general.ort_utils import ort_type_to_numpy


class MotionBertPoseConfig(BaseInferenceConfig):
    weights = "motion_bert_16.onnx"
    name: InferenceType = InferenceType.MOTIONBERT
    temporal_window = 10
    max_dynamic_batch = 20
    half: bool = True  # dont use fp 16



class MotionBert(InferenceWrapper):
    _config_type = MotionBertPoseConfig
    args: MotionBertPoseConfig
    input_name = "input"

    def __init__(self, msg_dict: Dict, is_local: Optional[bool] = None):
        super().__init__(msg_dict, is_local)
        self.image_size = msg_dict['image_size']
        

    @property
    def max_size_allowed(self):
        return {
            self.input_name: [self.args.max_dynamic_batch, self.args.temporal_window, 17, 3],
        }

    def _halpe2h36m(self, x):
        '''
            Input: x (T x V x C)  
            //Halpe 26 body keypoints
        {0,  "Nose"},
        {1,  "LEye"},
        {2,  "REye"},
        {3,  "LEar"},
        {4,  "REar"},
        {5,  "LShoulder"},
        {6,  "RShoulder"},
        {7,  "LElbow"},
        {8,  "RElbow"},
        {9,  "LWrist"},
        {10, "RWrist"},
        {11, "LHip"},
        {12, "RHip"},
        {13, "LKnee"},
        {14, "Rknee"},
        {15, "LAnkle"},
        {16, "RAnkle"},
        {17,  "Head"},
        {18,  "Neck"},
        {19,  "Hip"},
        {20, "LBigToe"},
        {21, "RBigToe"},
        {22, "LSmallToe"},
        {23, "RSmallToe"},
        {24, "LHeel"},
        {25, "RHeel"},
        '''
        T, V, C = x.shape
        y = np.zeros([T,17,C])
        y[:,0,:] = x[:,19,:]
        y[:,1,:] = x[:,12,:]
        y[:,2,:] = x[:,14,:]
        y[:,3,:] = x[:,16,:]
        y[:,4,:] = x[:,11,:]
        y[:,5,:] = x[:,13,:]
        y[:,6,:] = x[:,15,:]
        y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
        y[:,8,:] = x[:,18,:]
        y[:,9,:] = x[:,0,:]
        y[:,10,:] = x[:,17,:]
        y[:,11,:] = x[:,5,:]
        y[:,12,:] = x[:,7,:]
        y[:,13,:] = x[:,9,:]
        y[:,14,:] = x[:,6,:]
        y[:,15,:] = x[:,8,:]
        y[:,16,:] = x[:,10,:]
        return y

    def build_trt_model(self):
        from general.trt_utils import TrtInfer

        self.model = TrtInfer(self.args.weights, self.input_name, max_size_allowed=self.max_size_allowed)

    def build_full_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(self.args.weights, providers=["CUDAExecutionProvider"])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.args.half = ort_type_to_numpy[self.onnx_session.get_inputs()[0].type] is np.float16

        output_name_01 = self.onnx_session.get_outputs()[0].name
        self.output_names = [output_name_01]
        self.output_dict = {}
        for n in self.output_names:
            self.output_dict[n] = []

    def infer_full(self, skeletons: List[np.array]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skeletons_input = np.array(np.stack(skeletons))
            skeletons_lifted = self.onnx_session.run(
                    self.output_names, {self.input_name: skeletons_input.astype(self.data_type)}
                )
        return skeletons_lifted
    
    
    
    def prepare_crops(self, skeletons_list):
        skeletons_reduced = self._halpe2h36m(np.array(skeletons_list))
        w, h = self.image_size
        scale = min(w, h) / 2.0
        skeletons_reduced[:, :, :2] = skeletons_reduced[:, :, :2] - np.array([w, h]) / 2.0
        skeletons_reduced[:, :, :2] = skeletons_reduced[:, :, :2] / scale
    
        # Split skeletons_reduced into batches of 10 elements each
        num_batches = len(skeletons_reduced) // self.args.temporal_window
        skeleton_batches = [skeletons_reduced[i*self.args.temporal_window:(i+1)*self.args.temporal_window] for i in range(num_batches)]
        
        return skeleton_batches