from typing import List
import numpy as np
from numpy import ndarray
from skimage.morphology import thin, closing, square
from sklearn.cluster import KMeans
from math import floor

from general.analyzer_general import ROUTE_SHAPE
from general.entity import ActiveEntityData


class RouteSolver:
    binary_match = False

    def __init__(self):
        self._active_routes: List = []
        self._inactive_routes: List = []
        self._n_entities_period: int = 90
        self._n_time_period_sec: int = 5 * 60
        self._entities_per_route_pct: float = 4.0
        self._route_match_iou_th: float = 0.75
        self._route_sim_th: float = 0.5
        self._route_maps: List[ndarray] = []
        self._last_used_ts = -1
        self._kmeans_clusters = 10
        self._kmeans_occupancy_th = 0.15
        self._last_iteration_carry_sz = floor(self._n_entities_period * 0.5)
        self._last_iteration_carry = np.zeros((0, ROUTE_SHAPE[0] * ROUTE_SHAPE[1]))

    def extract_entity_route(self, entity: ActiveEntityData):
        route = entity.route_map()
        self._route_maps.append(route)

    @staticmethod
    def route_correlation_iou(route_1, route_2):
        and_result = np.bitwise_and(route_1, route_2)
        or_result = np.bitwise_or(route_1, route_2)
        iou_score = np.sum(and_result) / np.sum(or_result)
        return iou_score

    @staticmethod
    def route_correlation_L2(route_1, route_2):
        return np.correlate(route_1, route_2)[0]

    @staticmethod
    def match_routes_set(route_set_1, route_set_2, similarity_th: float = 0, binary_match=True):
        matches = np.zeros((len(route_set_1), len(route_set_2)))
        if binary_match:
            for i, entity_map in enumerate(route_set_1):
                # match all possible routes
                idmap = (entity_map > 0).astype(np.uint8)
                for j, route in enumerate(route_set_2):
                    matches[i, j] = RouteSolver.route_correlation_iou(idmap, route)
        else:
            route_set_1_norm = route_set_1 / np.linalg.norm(route_set_1, axis=1)[:, np.newaxis]
            route_set_2_norm = route_set_2 / np.linalg.norm(route_set_2, axis=1)[:, np.newaxis]
            for i, entity_map in enumerate(route_set_1_norm):
                # match all possible routes
                for j, route in enumerate(route_set_2_norm):
                    matches[i, j] = RouteSolver.route_correlation_L2(entity_map, route)

        matched_routes = np.argmax(matches, axis=1)
        matched_scores = np.amax(matches, axis=1)
        routes_count = np.bincount(matched_routes[matched_scores > similarity_th])
        matched_routes = np.where(matched_scores > similarity_th, matched_routes, -1)
        return matched_routes, routes_count

    def dedup_routes(self, routes):
        remaining_indices = list(range(len(routes)))
        out_routes = []
        morph_routes = []
        for route in routes:
            route = np.reshape(route, ROUTE_SHAPE)
            # Closing gaps in the image
            closed_image = closing(route, square(3))
            morph_routes.append(np.ravel(closed_image))
        while len(remaining_indices) > 0:
            curr_route = morph_routes[remaining_indices.pop(0)]
            dups = []
            for idx in remaining_indices:
                iou_th = RouteSolver.route_correlation_iou(curr_route, morph_routes[idx])
                if iou_th > self._route_sim_th:
                    dups.append(idx)

            if len(dups) > 0:
                dup_routes = [curr_route]
                for idx in dups:
                    dup_routes.append(morph_routes[idx])
                remaining_indices = [x for x in remaining_indices if x not in dups]
                matched_routes = np.stack(dup_routes)
                curr_route = np.mean(matched_routes, axis=0) >= 0.5
            out_routes.append(curr_route)
        return out_routes

    def match_routes(self, entity_maps: ndarray, routes, n_total_routes: int = None):
        # calculate matches and find hits
        matched_routes, routes_count = RouteSolver.match_routes_set(
            entity_maps, routes, self._route_match_iou_th, self.binary_match
        )

        new_routes = []
        end_points = []
        selected_indexes = np.full(len(matched_routes), -1)
        n_routes = n_total_routes if n_total_routes is not None else len(entity_maps)
        hit_th = self._entities_per_route_pct / 100 * n_routes
        for i, hit in enumerate(routes_count):
            if hit >= hit_th:
                selected_ents = np.where(matched_routes == i)
                new_routes.append(np.mean(entity_maps[selected_ents] > 0, axis=0) > 1 / hit)
                end_points.append(np.unravel_index(np.argmax(np.mean(entity_maps[selected_ents], axis=0)), ROUTE_SHAPE))
                selected_indexes[selected_ents] = i
            else:
                routes_count[i] = 0

        return routes_count, selected_indexes, new_routes, end_points

    def find_active_routes(self):
        n_routes = len(self._route_maps)
        if n_routes < self._n_entities_period:
            return []
        ents_routes = np.stack(self._route_maps, axis=0)
        ents_routes = np.concatenate((ents_routes, self._last_iteration_carry), axis=0).astype(int)
        found_routes = []
        end_points = []
        if not self.binary_match:
            ents_routes = ents_routes / np.amax(ents_routes, axis=1)[:, np.newaxis]

        if len(self._active_routes) > 0:
            routes_count, selected_indexes, new_routes, starts = self.match_routes(ents_routes, self._active_routes)
            ents_routes = ents_routes[selected_indexes < 0]
            inactive_routes = [self._active_routes[rid] for rid in range(len(routes_count)) if routes_count[rid] > 0]
            self._inactive_routes.append(inactive_routes)
            found_routes += new_routes
            end_points += starts

        if ents_routes.shape[0] > self._n_entities_period * 0.2:
            new_routes = self.search_for_new_routes(ents_routes)
            routes_count, selected_indexes, new_routes, starts = self.match_routes(ents_routes, new_routes, n_routes)
            ents_routes = ents_routes[selected_indexes < 0]
            found_routes += new_routes
            end_points += starts
        found_routes = self.dedup_routes(found_routes)
        self._last_iteration_carry = ents_routes[: self._last_iteration_carry_sz]
        self._route_maps.clear()
        self._active_routes = found_routes

        presentable_routes = []
        pad_width = 2
        # Thinning the image
        for i, route in enumerate(found_routes):
            route = np.pad(np.reshape(route, ROUTE_SHAPE), pad_width, "edge")
            # Closing gaps in the image
            closed = closing(route, square(5))
            thinned_image = np.asarray(thin(closed)[pad_width:-pad_width, pad_width:-pad_width], dtype=int)
            end_pt = end_points[i]
            thinned_image[end_pt[0], end_pt[1]] = 2
            presentable_routes.append(np.ravel(thinned_image))
        return presentable_routes

    def search_for_new_routes(self, entity_maps: ndarray, n_routes: int = None):
        if n_routes is None:
            n_routes = self._kmeans_clusters
        kmeans = KMeans(n_clusters=n_routes, init="k-means++", max_iter=500, n_init=10)
        # kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=500, n_init=10)
        # kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, max_iter=500, n_init=1)
        if self.binary_match:
            kmeans.fit((entity_maps > 0).astype(float))
            cluster_centers = (kmeans.cluster_centers_ > self._kmeans_occupancy_th).astype(int)
        else:
            kmeans.fit(entity_maps)
            cluster_centers = kmeans.cluster_centers_
        # clusters_labels = kmeans.labels_
        return cluster_centers
