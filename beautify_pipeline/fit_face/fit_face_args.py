class FitFaceArgs():
    def __init__(
            self,
            src_file,
            face_path,
            mask_path,
            face_landmarks_path,
            dst_file
        ):
        self.src_file = src_file
        self.face_path = face_path
        self.mask_path = mask_path
        self.face_landmarks_path = face_landmarks_path
        self.dst_file = dst_file
