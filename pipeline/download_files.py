import requests
from multiprocessing.pool import ThreadPool
from pathlib import Path
import os
import zipfile
 
def download_url(url, directory="data/allbooks"):
    print("downloading: ",url)
    # assumes that the last segment after the / represents the file name
    # if url is abc/xyz/file.txt, the file name will be file.txt
    file_name_start_pos = url.rfind("/") + 1
    file_name = url[file_name_start_pos:]

    save_location = Path(directory)
    save_location = save_location / file_name

    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        print(f"saving to {save_location}")
        with open(save_location, 'wb') as f:
            for data in r:
                f.write(data)
    return url

def unzip_all(directory="data/allbooks"):
    unzip_dir = Path(directory)
    extension = ".zip"
    for item in os.listdir(unzip_dir): # loop through items in dir

        if item.endswith(extension): # check for ".zip" extension
            file_name = unzip_dir / item # get full path of files
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(unzip_dir) # extract file to dir
            zip_ref.close() # close file
            os.remove(file_name) # delete zipped file
            print(f"{file_name} unpacked")

if __name__ == "__main__":


    from string import ascii_uppercase

    urls = []
    for letter in ascii_uppercase:
        urls.append(f"https://archive.org/download/Fanfictiondotnet1011dump/Fanfiction_{letter}.zip")
        
    
    # urls = ["https://jsonplaceholder.typicode.com/posts",
    #         "https://jsonplaceholder.typicode.com/comments",
    #         "https://jsonplaceholder.typicode.com/photos",
    #         "https://jsonplaceholder.typicode.com/todos",
    #         "https://jsonplaceholder.typicode.com/albums"
    #         ]
    
    # Run 5 multiple threads. Each call will take the next element in urls list
    results = ThreadPool(len(urls)).imap_unordered(download_url, urls)
    for r in results:
        print(r)

    print("unzipping files now")

    unzip_all()

