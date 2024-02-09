from interfaces import Tracker

class BotSort(Tracker):
  def __init__(self):
    super().__init__()
    self.tracker = None

  def __str__(self):
    return "botsort.yaml"

  def update(self, detections):
    return detections