import cv2
import numpy as np

class VisualOdometry:
    def __init__(self):
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.prev_kp = None
        self.prev_des = None
        self.prev_frame = None
        
        # Rough estimate of camera pose (3D translation + orientation)
        self.pose = np.eye(4, dtype=np.float32)  # 4x4 transformation matrix

        # BFMatcher for feature matching
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_frame(self, frame):
        """
        Estimate the drone's motion by comparing features with the previous frame.
        Returns updated pose and debug frame showing matched features.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        debug_frame = frame.copy()

        if self.prev_frame is None:
            # First frame, just set reference
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose, debug_frame

        # Match descriptors
        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Use only good matches (for demonstration)
        good_matches = matches[:50]
        
        # Extract matched keypoints
        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute essential matrix (assuming a known focal length and principal point)
        # For demonstration, let's assume a dummy focal length and principal point
        focal = 700.0
        pp = (frame.shape[1] / 2, frame.shape[0] / 2)
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is not None:
            # Recover relative camera rotation and translation
            _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, focal=focal, pp=pp)

            # Update transformation matrix (this is a simplistic approach)
            transformation = np.eye(4, dtype=np.float32)
            transformation[:3, :3] = R
            transformation[:3, 3] = t.reshape(-1)

            self.pose = self.pose @ transformation

        # Debug: draw matched keypoints
        debug_frame = cv2.drawMatches(self.prev_frame, self.prev_kp, gray, kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Update previous frame data
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des

        return self.pose, debug_frame
