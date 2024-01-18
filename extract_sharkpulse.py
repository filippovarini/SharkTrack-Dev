import requests
import csv
import os

# Function to download images
def download_images(image_names, base_url):
    valid_extensions = ['jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'PNG']
    output_dir = '../images_raw/sharkpulse/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name in image_names:
        downloaded = False
        for ext in valid_extensions:
            image_url = f"{base_url}{name}.{ext}"
            response = requests.get(image_url, stream=True)
            
            if response.status_code == 200:
                with open(os.path.join(output_dir, f"{name}.{ext}"), 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"Downloaded: {name}.{ext}")
                downloaded = True
                break  # Break loop if image is downloaded successfully
        
        if not downloaded:
            print(f"Failed to download: {name}")

# Path to your CSV file with image names (without extensions)
csv_file_path = '../annotations/sharkpulse.csv'

# Column name in the CSV file containing image names
column_name = 'img_name'

# URL of the online directory where images are located
base_url = 'http://sp2.cs.vt.edu:3838/SDvalidation_2/shark/'

# Read image names from CSV file
image_names = []
with open(csv_file_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_names.append(row[column_name])

downloaded = ['.'.join(img.split('.')[:-1]) for  img in os.listdir('../images_raw/sharkpulse/')]

image_names = list(set(image_names) - set(downloaded))


# Download images
download_images(image_names, base_url)