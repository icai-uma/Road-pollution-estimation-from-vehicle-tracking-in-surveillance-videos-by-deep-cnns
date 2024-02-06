import numpy as np 
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
import math
from Config import Config

def get_association_cost_matrix(tracks, detections, dist_threshold):
    num_of_traces = len(tracks)
    cost = []                                           # This will contain the cost matrix.
    traces_with_possible_detection_to_associate = []    # Traces that has a detection with cost lesser than self.dist_threshold

    for trace_index in range(num_of_traces):
        #For each trace, we get the cost of it prediction to the detections and append it to the cost matrix.
        #The resulting mattix will have the traces as rows and the columns as detection related costs.

        diff1 = np.linalg.norm(tracks[trace_index].prediction - detections.reshape(-1,2), axis=1)
        diff2 = np.sqrt(np.sum(np.power(tracks[trace_index].prediction - detections.reshape(-1,2),2), axis=1))
        diff = diff2 + (diff2 > dist_threshold) * Config.TRACKER_VALUE_TO_USE_AS_INF
        if np.sum(diff < dist_threshold):
            traces_with_possible_detection_to_associate.append(trace_index)
        cost.append(diff)

    if len(cost) == 0: cost.append([])          # This line is to avoid an error in linear_sum_assignment if there is no detection and no traces.
    
    cost = np.array(cost)

    return cost, traces_with_possible_detection_to_associate

def get_more_skipped_trace_index(tracks, un_assigned_tracks_index):

    more_skipped_track_index = -1
    max_skips = -1
    for track_index in un_assigned_tracks_index:
        if tracks[track_index].skipped_frames > max_skips:
            max_skips = tracks[track_index].skipped_frames
            more_skipped_track_index = track_index

    return more_skipped_track_index


class Tracks(object):
    """docstring for Tracks"""
    def __init__(self, detection, trackId, maxlen=200):
        super(Tracks, self).__init__()
        self.KF = KalmanFilter(initial_state =[[detection[0]], [0], [detection[1]], [0]])
        self.KF.predict()
        self.KF.correct(np.matrix(detection).reshape(2,1))
        self.trace = deque(maxlen=maxlen)
        self.prediction = detection.reshape(1,2)
        self.trackId = trackId
        self.skipped_frames = 0

    def predict(self,detection):
        self.prediction = np.array(self.KF.predict()).reshape(1,2)
        self.KF.correct(np.matrix(detection).reshape(2,1))


class Tracker(object):
    """docstring for Tracker"""
    def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
        super(Tracker, self).__init__()
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.max_trace_length = max_trace_length
        self.trackId = 0
        self.tracks = []

    def update(self, detections):

        associations = {}
        predictions = {}
        costs = {}

        """
        if len(self.tracks) == 0:
            for i in range(detections.shape[0]):
                track = Tracks(detections[i], self.trackId)
                self.trackId +=1
                self.tracks.append(track)
        """
        
        cost, _ = get_association_cost_matrix(self.tracks, detections, self.dist_threshold)
        """
        print(len(self.tracks))
        print(cost)
        print("--")
        """
        #cost = np.array(cost)*0.1
        row, col = linear_sum_assignment(cost)  # We get the associations with the minimum cost.

        num_traces = len(self.tracks)
        trace_assignment = np.array([-1]*num_traces)
        for i in range(len(row)):
            # For each association, we assign the detection index to the trace index to get the minimum global cost.
            trace_assignment[row[i]] = col[i]

        un_assigned_tracks_index = []
        un_assigned_tracks = []
        print(trace_assignment)
        for i in range(trace_assignment.shape[0]):
            # For each trace index, we will unmade the asociation if the cost is greater than self.dist_threshold

            if trace_assignment[i] != -1:
                association_cost = cost[i][trace_assignment[i]]

                if Config.TRACKER_DEBUG: print(f"{i}th track with id {self.tracks[i].trackId} has cost {association_cost}.")

                if (association_cost > self.dist_threshold):
                    if Config.TRACKER_DEBUG: print("We consider this trace as not associated due high cost.")
                    trace_assignment[i] = -1
                    un_assigned_tracks_index.append(i)
                    un_assigned_tracks.append(self.tracks[i])

                else:
                    costs[self.tracks[i].trackId] = association_cost
            else:
                un_assigned_tracks_index.append(i)
                un_assigned_tracks.append(self.tracks[i])
                if Config.TRACKER_DEBUG: print(f"{i}th track with id {self.tracks[i].trackId} was not associated.")

        del_tracks = []
        for i in un_assigned_tracks_index:
            # For each un asigned trace index, we will increrase its skip frame counter and set it to be deteled if it has been skipped more than self.max_frame_skipped.
            self.tracks[i].skipped_frames +=1
            if Config.TRACKER_DEBUG: print(f"{i}th track with id {self.tracks[i].trackId} has {self.tracks[i].skipped_frames} skipped frames.")
            if self.tracks[i].skipped_frames > self.max_frame_skipped:
                if Config.TRACKER_DEBUG: print(f"The {i}th track must be deleted.")
                del_tracks.append(i)

        """
        if len(del_tracks) > 0:
            for i in range(len(del_tracks)):
                del self.tracks[i]
                del trace_assignment[i]
        """
        
        for i in range(len(trace_assignment)):
            if(trace_assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                print(f"Track with id {self.tracks[i].trackId} predicts before update {self.tracks[i].prediction}")
                predictions[self.tracks[i].trackId] = self.tracks[i].prediction
                self.tracks[i].predict(detections[trace_assignment[i]])
                print(f"Track with id {self.tracks[i].trackId} predicts after update {self.tracks[i].prediction}")
                associations[self.tracks[i].trackId] = detections[trace_assignment[i]]    
            else:
                print(f"Track with id {self.tracks[i].trackId} predicts {self.tracks[i].prediction}")
                predictions[self.tracks[i].trackId] = self.tracks[i].prediction
                self.tracks[i].predict(self.tracks[i].prediction)
                associations[self.tracks[i].trackId] = None
                costs[self.tracks[i].trackId] = None
            self.tracks[i].trace.append(self.tracks[i].prediction)
        
        if len(del_tracks) > 0:
            del_tracks.reverse()
            for i in del_tracks:                
                #print(trace_assignment)
                #print(f"The {i}th track with id {self.tracks[i].trackId} has been deleted.")
                track_to_delete = self.tracks.pop(i)                
                del track_to_delete
                #print(trace_assignment)
                
        for i in range(len(detections)):
            if i not in trace_assignment:
                print(f"We create a track to follow the detection {i} with track id {self.trackId}.")
                track = Tracks(detections[i], self.trackId)
                associations[track.trackId] = detections[i]
                self.trackId +=1
                self.tracks.append(track)
                predictions[track.trackId] = track.prediction
                costs[track.trackId] = 0

        """
        print("associations")
        print(associations)
        print("predictions")
        print(predictions)
        print("costs")
        print(costs)
        """
        print(associations)
        print(predictions)
        return associations, predictions, costs
