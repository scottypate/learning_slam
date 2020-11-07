import numpy as np

import cv2
import sdl2
import sdl2.ext


class Display:
    """
    Wrapper library to create a quick video display using cv2 and sdl2

    Args:
        width(int): The video width in pixels
        height(int): The video height in pixels

    Returns:
        None
    """

    def __init__(self, width, height):
        sdl2.ext.init()
        self.width, self.height = width, height
        self.window = sdl2.ext.Window(
            title="SLAM", size=(width, height), position=(800, 500)
        )
        self.window.show()
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher()
        self.last = None

    def process_frame(self, frame):
        """
        Resize the frame from the raw size to the given width and height

        Args:
            frame(numpy.ndarray): The frame from cv2

        Returns:
            numpy.ndarray: The resized frame
        """
        return cv2.resize(frame, (self.width, self.height))

    def find_orbs(self, frame):
        """
        Find and plot the orbs in the frame
        https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html

        Args:
            frame(numpy.ndarray): The frame from cv2

        Returns:
            numpy.ndarray: The resized frame
        """
        features = cv2.goodFeaturesToTrack(
            image=np.mean(frame, axis=2).astype(np.uint8),
            maxCorners=3000,
            qualityLevel=0.01,
            minDistance=3,
        )
        keypoints = []

        for feature in features:
            u, v = map(lambda x: int(round(x)), feature[0])
            cv2.circle(img=frame, center=(u, v), color=(0, 255, 0), radius=2)
            keypoint = cv2.KeyPoint(x=feature[0][0], y=feature[0][1], _size=20)
            keypoints.append(keypoint)

        return self.orb.compute(frame, keypoints)

    def find_matches(self, descriptors):
        """
        Use the Orb descriptors to match discovered features between
        frames in the video

        Args:
            descriptors(numpy.ndarray): The descriptors for the current frame

        Returns:
            list[cv2.DMatch]: A list of the discovered matches
        """
        if self.last:
            good_matches = []
            matches = self.matcher.knnMatch(descriptors, self.last["descriptors"], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])

            return good_matches

        else:
            return []

    def draw(self, frame):
        """
        Open a display window and draw the frames

        Args:
            frame(numpy.ndarray): The frame from cv2

        Returns:
            None
        """
        resized_frame = self.process_frame(frame)
        keypoints, descriptors = self.find_orbs(resized_frame)
        matches = self.find_matches(descriptors)
        self.last = {"keypoints": keypoints, "descriptors": descriptors}
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)
        surface = sdl2.ext.pixels3d(self.window.get_surface())
        surface[:, :, 0:3] = resized_frame.swapaxes(0, 1)
        self.window.refresh()
