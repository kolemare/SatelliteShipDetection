import os
import zipfile

def combine_and_extract(part1_path, part2_path, extract_to="training/dataset/ships_in_satellite_imagery"):
    combined_zip = "combined_temp.zip"

    # Combine the two parts
    with open(combined_zip, 'wb') as out_file:
        for part in [part1_path, part2_path]:
            with open(part, 'rb') as p:
                out_file.write(p.read())

    # Extract the combined zip
    with zipfile.ZipFile(combined_zip, 'r') as zip_ref:
        os.makedirs(extract_to, exist_ok=True)
        zip_ref.extractall(extract_to)

    os.remove(combined_zip)
    print(f"✅ Extraction complete to '{extract_to}/'.")

def validate_parts(part1, part2):
    return os.path.exists(part1) and os.path.exists(part2)
