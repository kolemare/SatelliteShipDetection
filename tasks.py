from invoke import task
from tools import combine_and_extract, validate_parts

@task
def extract(c):
    """
    Extracts split zip parts into the 'dataset/ships_in_satellite_imagery' folder.
    Run with: invoke extract
    """
    part1 = "dataset/ships_in_satellite_imagery_part1.zip"
    part2 = "dataset/ships_in_satellite_imagery_part2.zip"
    extract_dir = "dataset/ships_in_satellite_imagery"

    if validate_parts(part1, part2):
        combine_and_extract(part1, part2, extract_dir)
    else:
        print("❌ Missing one or both zip parts. Please check the files.")
