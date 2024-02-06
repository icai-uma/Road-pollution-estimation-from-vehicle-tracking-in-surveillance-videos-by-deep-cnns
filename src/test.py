from tracker import Tracker
import numpy as np

tracker = Tracker(100, 20, 30)

associations, predictions, costs = tracker.update(np.array([[-1,-1]]))
print("------------------------------------------------------------------")
associations, predictions, costs = tracker.update(np.array([[-10,-10]]))
print("------------------------------------------------------------------")
associations, predictions, costs = tracker.update(np.array([[-20,-20]]))
print("------------------------------------------------------------------")
associations, predictions, costs = tracker.update(np.array([[-30,-30]]))
print("------------------------------------------------------------------")
associations, predictions, costs = tracker.update(np.array([[-40,-40]]))
print("------------------------------------------------------------------")
associations, predictions, costs = tracker.update(np.array([[-50,-50]]))
print("------------------------------------------------------------------")
associations, predictions, costs = tracker.update(np.array([[-60,-60]]))
print("------------------------------------------------------------------")
associations, predictions, costs = tracker.update(np.array([[-70,-70]]))
