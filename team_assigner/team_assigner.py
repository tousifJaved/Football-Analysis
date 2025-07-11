from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None  # store the KMeans model after fitting

    def get_clustering_model(self, image):
        # Reshape the image to 2D array (pixels x 3 color channels)
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Crop the bounding box area from frame
        x1, y1, x2, y2 = map(int, bbox)
        image = frame[y1:y2, x1:x2]

        # Handle empty or invalid crop
        if image.size == 0:
            return np.array([0, 0, 0])  # fallback color

        # Take top half of the cropped image to focus on jersey
        top_half_image = image[: image.shape[0] // 2, :]

        # Get clustering model for jersey color clusters
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels reshaped to image dimensions
        labels = kmeans.labels_
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        # Find the most common cluster at corners (assumed background/non-player)
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Player cluster is the opposite cluster
        player_cluster = 1 - non_player_cluster

        # Return the RGB center color of the player cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Fit KMeans to player colors to identify two team clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        # Store team colors for reference
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # Return cached team if assigned before
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Predict team based on player color
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1  # 1 or 2

        # Cache the assignment
        self.player_team_dict[player_id] = team_id

        return team_id
