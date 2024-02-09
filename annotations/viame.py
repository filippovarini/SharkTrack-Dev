from datetime import timedelta, datetime
import pandas as pd

def format_time(seconds):
    """
    Formats seconds to mm:ss:ms
    if 0 m, show 00 instead of 0
    ms should only be 2 digits
    """
    # Convert seconds to a timedelta
    td = timedelta(seconds=seconds)

    # Extract minutes, seconds, and milliseconds
    minutes = int(td.total_seconds() // 60)
    seconds = int(td.total_seconds() % 60)
    milliseconds = int(td.microseconds / 10000)  # Convert microseconds to milliseconds and round to 2 digits

    # Format the time string
    time_str = f"{minutes:02}:{seconds:02}:{milliseconds:02}"
    return time_str

def track_history_to_viame(track_history, track_fps):
  """
  Input:
    - track_history: dataframe with columns: track_id,frame_id,confidence,bbox_xyxy
    - track_fps: int, fps of the track
    - original_fps: int, fps of the original video

  Creates annotation df respecting required format:
  - columns: '# 1: Detection or Track-id',	'2: Video or Image Identifier' '3: Unique Frame Identifier'	'4-7: Img-bbox(TL_x'	'TL_y'	'BR_x'	'BR_y)'	'8: Detection or Length Confidence'	'9: Target Length (0 or -1 if invalid)'	'10-11+: Repeated Species	Confidence Pairs or Attributes'
  - rows: one row per bbox, ordered per track_id and frame. So first all bboxes for track_id 1 in frame order, then track_id 2, etc.
"""
  # Create df
  columns = ['# 1: Detection or Track-id',	'2: Video or Image Identifier', '3: Unique Frame Identifier',	'4-7: Img-bbox(TL_x',	'TL_y',	'BR_x',	'BR_y)',	'8: Detection or Length Confidence',	'9: Target Length (0 or -1 if invalid)',	'10-11+: Repeated Species', 'Confidence Pairs or Attributes']
  df = pd.DataFrame(columns=columns)

  for index, row in track_history.iterrows():
    time = int(row['frame_id'] / track_fps)
    time = format_time(time)

    # Add row
    new_row = {
      '# 1: Detection or Track-id': row['track_id'], 
      '2: Video or Image Identifier': time, 
      '3: Unique Frame Identifier': row['frame_id'], 
      '4-7: Img-bbox(TL_x': row['bbox_xyxy'][0], 
      'TL_y': row['bbox_xyxy'][1], 
      'BR_x': row['bbox_xyxy'][2], 
      'BR_y)': row['bbox_xyxy'][3], 
      '8: Detection or Length Confidence': row['confidence'],
      '9: Target Length (0 or -1 if invalid)': -1,
      '10-11+: Repeated Species': 'shark',
      'Confidence Pairs or Attributes': 1
      }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
          
  # Sort by track_id and frame_id
  df = df.sort_values(by=['# 1: Detection or Track-id', '3: Unique Frame Identifier'])
  df = df[columns]

  # Add a line below the header with the values '# metadata'	'fps: 1'	'exported_by: "dive:python"', 'exported_time: "Mon Jan 15 15:56:15 2024"'	'Unnamed: 4'	'Unnamed: 5'	'Unnamed: 6'	'Unnamed: 7'	'Unnamed: 8'	'Unnamed: 9'	'Unnamed: 10'
  current_date = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
  metadata_row = {'# 1: Detection or Track-id': '# metadata', '2: Video or Image Identifier': f'fps: {track_fps}', '3: Unique Frame Identifier': 'exported_by: "Filippo Varini"', '4-7: Img-bbox(TL_x': f'exported_time: "{current_date}"', 'TL_y': 'Unnamed: 4', 'BR_x': 'Unnamed: 5', 'BR_y)': 'Unnamed: 6', '8: Detection or Length Confidence': 'Unnamed: 7', '9: Target Length (0 or -1 if invalid)': 'Unnamed: 8', '10-11+: Repeated Species': 'Unnamed: 9', 'Confidence Pairs or Attributes': 'Unnamed: 10'}
  df = pd.concat([df, pd.DataFrame([metadata_row])], ignore_index=True)
  # make sure metadata row is at the beginning of the df (index 0)
  df = df.reindex([len(df)-1] + list(range(len(df)-1)))
  # remove indices
  df = df.reset_index(drop=True)

  return df
   
   