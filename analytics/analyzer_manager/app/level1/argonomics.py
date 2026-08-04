import numpy as np
from general.body_part_reba_calculator import Pose_to_degrees as reba_transformer
from general.body_part_reba_calculator import Degree_to_REBA as REBA

class RebaScore():
    body_part_numbers = [2, 18, 3, 13, 4, 5, 12, 11, 17, 7, 16, 8, 9, 15, 14, 0, 1]
    body_part_names = ["Head", "Hips", "LeftArm", "LeftFoot", "LeftForeArm", "LeftHand", "LeftLeg", "LeftUpLeg", "Neck","RightArm", "RightFoot", "RightForeArm", "RightHand", "RightLeg", "RightUpLeg", "Spine", "Spine1"]
    body_part_numbers_coco = [10, 0, 11, 6, 12, 13, 5, 4, 8, 14, 3, 15, 16, 2, 1, 7, 8]            
    scales = [12, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # Specify the scale for each score
    # def __init__(self):
        
    def get_score(self, skeletons):
        rebas_for_frame = []
        for idx_in_clip, frame_id in enumerate(skeletons):
            k_3d = skeletons[idx_in_clip]  # [17, 3]

            # Prepare data for REBA computation
            keypoint_3d_transformed = np.zeros((19, 3))

            # Map the keypoints accordingly
            spine1_3d = (k_3d[11, :] + k_3d[14, :]) / 2  # Upper spine as 3D point between shoulders
            keypoint_3d_transformed[RebaScore.body_part_numbers, :] = k_3d[RebaScore.body_part_numbers_coco, :]
            keypoint_3d_transformed[1, :] = spine1_3d

            # Compute joint angles
            m_transformer = reba_transformer.PoseToDeg(keypoint_3d_transformed, np.zeros((19, 4)))
            joints_degrees = m_transformer.degree_computation()

            # Compute REBA scores
            m_reba = REBA.DegreeToREBA(joints_degrees)
            REBA_scores = m_reba.reba_computation()
            rebas_for_frame.append(REBA_scores)
            
        return REBA_scores

    def get_level(self):
        if self.score < 2:
            return "Low"
        elif self.score < 4:
            return "Moderate"
        elif self.score < 7:
            return "High"
        else:
            return "Very High"

    def get_description(self):
        if self.score < 2:
            return "Low risk of injury"
        elif self.score < 4:
            return "Moderate risk of injury"
        elif self.score < 7:
            return "High risk of injury"
        else:
            return "Very high risk of injury"

    def get_recommendation(self):
        if self.score < 2:
            return "No action required"
        elif self.score < 4:
            return "Consider changes"
        elif self.score < 7:
            return "Change required"
        else:
            return "Immediate action required"

    def get_color(self):
        if self.score < 2:
            return "green"
        elif self.score < 4:
            return "yellow"
        elif self.score < 7:
            return "orange"
        else:
            return "red"