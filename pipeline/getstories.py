# with fanfiction scraper
# from fanfiction import Scraper
# scraper = Scraper()
# storydict = scraper.scrape_story( 13395849,keep_html = False)

# with open("storyfile.txt", "w") as text_file:
#     import pdb; pdb.set_trace()
#     # text_file.write(storydict["chapters"][1].decode('utf-8'))
#     # text_file.write(storydict["chapters"][2].decode('utf-8'))

import os
from joblib import Memory
from pathlib import Path
import random

cachedir = "cache"
memory = Memory(cachedir, verbose=10)



def get_text_paths(datadir, shuffle = False):

    files = []
    # r=root, d=directories, f = files
    path = Path("pipeline/data")
    path = path /  datadir
    
    for r,d, f in os.walk(path):

        for fl in f:
            
            if '.txt' in fl:
                files.append(os.path.join(r, fl))
    
    
    if shuffle:
        random.shuffle(files)
        print("shuffled files")

    return files

# @memory.cache
def parse_fanfiction(files):
    books = []
    # files = files[0:2]
    for fileid, file in enumerate(files):
        if fileid % 1000 == 0:
            print(f"parse book {fileid} now")
        bookdata = {"book_id": fileid }
        current_chapter = -1
        with open(file, "r", encoding="utf-8") as openedfile:
            lines = openedfile.readlines()
            # hello = openedfile.read()
            
            for lid, line in enumerate(lines[:-1]):
                line = line.replace("\n", "")
                if lid == 3:
                    bookdata["Title"] = line
                if lid == 5:
                    bookdata["Author"] = line
                if  23 > lid > 8:
                    split_id = line.find(":")
                    key = line[:split_id]
                    value = line[split_id+2:]
                    bookdata[key] = value
                
                if lid > 26:
                    if line[:1] == "\t":
                        current_chapter += 1
                        bookdata[current_chapter] = {"chapter_headline":line[1:], "content": []}
                    else:
                        if current_chapter >= 0:
                            try:
                                bookdata[current_chapter]["content"].append(line)
                            except:
                                import pdb; pdb.set_trace()
            bookdata[current_chapter]["content"].append("")
            bookdata[current_chapter]["content"].append("THE END")

            books.append(bookdata)
                    




    return books
            # bookdata[""]




# with open("teststory.txt", "r") as text_file:
#     tt = text_file.read()
#     print(tt)




if __name__ == "__main__":
    path = "allbooks"
    books = get_ff_files(path)
    print(len(books))
    print(books[-1])

