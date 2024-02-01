# Define annotation interface with ABC

from abc import ABC, abstractmethod

class Annotation(ABC):
  def get_bbox(self, row):
    """get dataframe row and return bbox"""
    pass

  def get_id(self, row):
    """get dataframe row and return id
    ID format: source_imageID
    """
    pass

  def get_taxonomy(self, row):
    """get dataframe row and return family, genus, species"""
    pass