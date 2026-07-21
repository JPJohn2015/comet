# python imports
import json
from pathlib import Path
import os
import requests
import time
import certifi

# comet imports
from comet.utilities.constants import Constants as c

class DataManager:
    """Class that Processes, Downloads and Updates data files.

    Example Constructions:
        * propagator = Propagator(epoch, state)
    """
    def __init__(self):
        # Path to Data Folder
        self._path = Path(__file__).parents[1]

        # Load Download Paths JSON
        f = open(self._path / 'download_paths.json')
        self._download_paths = json.load(f)
        
    def update_files(self):
        for file in self._download_paths:
            # Get file path and determine when it was last modified
            file_path = self._path / file['folder'] / file['file']
            since_modified = (time.time() - file_path.stat().st_mtime)/c.DAY
            
            # Check if file needs to be updated
            match file['update']:
                case 'daily':
                    need_update = since_modified >= 1
                case 'weekly':
                    need_update = since_modified >= 7
                case 'monthly':
                    need_update = since_modified >= 30
                case _:
                    need_update = False

            # Update files as needed
            if need_update:
                # Download new file
                print(f'{file["file"]} out of date, downloading new version.')
                response = requests.get(file['source'], verify=certifi.where())
                if (not response.ok) or (response.status_code != 200):
                    print(f'{file["file"]} download failed.')
                    continue
                with open(file_path, mode="wb") as file:
                    file.write(response.content)
                print(f'{file["file"]} downloaded successfully.')

# Testing
if __name__ == "__main__":

    dm = DataManager()
    dm.update_files()
