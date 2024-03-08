EVENTMEASURE_COLUMNS = ['Filename', 'Frame', 'Time..mins.', 'Period', 'Period.time..mins.', 'Image.row', 'Image.col', 'Width', 'Height', 'OpCode', 'TapeReader', 'Depth', 'Comment', 'Location_Name', 'Deploy_Year', 'Deploy_Month', 'Deploy_Day', 'Local_Time', 'GPS_Latitude', 'GPS_Longitude',	'Habitat_Code',	'Family',	'Genus', 'Species',	'Code', 'Number', 'Stage'	Activity	Comment.1	X	X.1 ]

def output2eventmeasure(track_results):
  for chapter_id, chapter_results in track_results.items():
    for frame_id, frame_results in enumerate(chapter_results):
