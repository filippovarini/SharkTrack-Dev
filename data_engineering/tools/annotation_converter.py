import pandas as pd
import os


SHARKTRACK_COLUMNS = ['filename', 'class', 'ymin', 'xmin', 'xmax', 'ymax', 'source', 'track_id', 'frame_id']

def categorise_species(species):
  # TODO: filter out behaviours
  return 'shark or ray'

def viame2standard(csv_path, source, download_dir, annotations_fps, is_video):
    assert annotations_fps in [1,10], "Only 1 or 10 fps are supported"
    # Load the CSV file

    df = pd.read_csv(csv_path)
    if df.head(1).values.tolist()[0][0].startswith('#'):
        df = pd.read_csv(csv_path, skiprows=lambda x: x in [1]) # skip row if metadata

    data = []

    for _, row in df.iterrows():
        # Build the Filename
        track_id = int(row["# 1: Detection or Track-id"])
        frame_id = int(row['3: Unique Frame Identifier'])

        if annotations_fps == 10 and frame_id % 10 != 0:
            # extract only 1fps annotations
            continue
    
        frame_id = int(frame_id / annotations_fps) # id starts at 0 and increments by 1 (if fps 1 doesn't change)

        if is_video:
            filename = f"{source}_frame{frame_id}.jpg"
        else:
            filename = row['2: Video or Image Identifier']
        img_path = os.path.join(download_dir, source, filename)
        assert os.path.exists(img_path), f"File {img_path} does not exist"

        species = row["10-11+: Repeated Species"]
        label = categorise_species(species)
        # TODO: filter out behaviour classification and clean only to shark/ray
        
        xmin = row['4-7: Img-bbox(TL_x']
        ymin = row['TL_y']
        xmax = row['BR_x']
        ymax = row['BR_y)']

        if not filename.startswith(source):
            filename = f"{source}_{filename}" 
        
        # Prepare the new row as a Series
        row = {
            'filename': filename,
            'class': label,
            'ymin': ymin,
            'xmin': xmin,
            'xmax': xmax,
            'ymax': ymax,
            'source': source,
            'track_id': track_id,
            'frame_id': frame_id,
        }

        data.append(row)

    converted_df = pd.DataFrame(data, columns=SHARKTRACK_COLUMNS)

    return converted_df