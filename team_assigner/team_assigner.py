import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, deque


class TeamAssigner:
    def __init__(self, smoothing_window=15):
        self.team_colors = {}
        self.kmeans = None
        self.player_history = defaultdict(lambda: deque(maxlen=smoothing_window))

    # Performs Kmeans on AB channels
    def get_clustering_model(self, image):
        if image.size == 0:
            return None, None
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(image_lab)
        mask = (L > 30) & (L < 220)
        ab = np.stack((A, B), axis=-1)
        ab_masked = ab[mask]
        if len(ab_masked) < 50:
            return None, None
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(ab_masked)
        return kmeans, mask.shape

    # Identify background clusters
    def filter_background(self, labels_2d):
        h, w = labels_2d.shape
        border_pixels = np.concatenate(
            [
                labels_2d[0:5, :].flatten(),
                labels_2d[-5:, :].flatten(),
                labels_2d[:, 0:5].flatten(),
                labels_2d[:, -5:].flatten(),
            ]
        )
        border_pixels = border_pixels[border_pixels >= 0]
        if len(border_pixels) == 0:
            return 0
        counts = np.bincount(border_pixels)
        return np.argmax(counts)

    def ab_to_rgb(self, ab):
        lab_color = np.zeros((1, 1, 3), dtype=np.uint8)
        lab_color[0, 0, 0] = 150
        lab_color[0, 0, 1:] = ab.astype(np.uint8)
        rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0, 0]
        return rgb_color

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return np.array([0, 0, 0])
        image = image[: image.shape[0] // 2, :]
        kmeans, mask_shape = self.get_clustering_model(image)
        if kmeans is None:
            return np.array([0, 0, 0])
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = image_lab[:, :, 0]
        mask = (L > 30) & (L < 220)
        labels_2d = np.full(L.shape, -1)
        labels_2d[mask] = kmeans.labels_
        background_cluster = self.filter_background(labels_2d)
        player_cluster = 1 - background_cluster
        player_color_ab = kmeans.cluster_centers_[player_cluster]
        return self.ab_to_rgb(player_color_ab)

    # Performs Kmean clustering, stores most 2 dominant colors and their centeroids
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, detection in player_detections.items():
            bbox = detection["bbox"]
            color = self.get_player_color(frame, bbox)
            player_colors.append(color)
        if len(player_colors) < 2:
            return
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id=None):
        if self.kmeans is None:
            return None
        color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(color.reshape(1, -1))[0] + 1
        if player_id is not None:
            self.player_history[player_id].append(team_id)

            # Return the most frequent team in the window
            team_votes = list(self.player_history[player_id])
            return max(set(team_votes), key=team_votes.count)
        return team_id
