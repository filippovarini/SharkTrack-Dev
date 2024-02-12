from interface import Annotation

class Viame(Annotation):
  """Class for VIAME annotation files"""
  def get_bbox(self, row):
    xmin = row['4-7: Img-bbox(TL_x']
    ymin = row['TL_y']
    xmax = row['BR_x']
    ymax = row['BR_y)']

    return [xmin, ymin, xmax, ymax]
  
  def get_frame_id(self, row):
    return int(row['3: Unique Frame Identifier'])
  
  def get_track_id(self, row):
    return row["# 1: Detection or Track-id"]
  

  def get_taxonomy(self, row):
    species_parts = row["10-11+: Repeated Species"].split(" ")
    genus = species_parts[0]
    species = species_parts[1] if len(species_parts) > 1 else ""
    return "", genus, species