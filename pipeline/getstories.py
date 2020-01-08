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

cachedir = "cache"
memory = Memory(cachedir, verbose=10)



def get_text_paths(datadir):

    files = []
    # r=root, d=directories, f = files
    datadir = os.getcwd() + "\pipeline\data\\" + datadir
    
    for r,d, f in os.walk(datadir):

        for fl in f:
            
            if '.txt' in fl:
                files.append(os.path.join(r, fl))
    
    
    return files

@memory.cache
def parse_fanfiction(files):
    books = []
    # files = files[0:2]
    for fileid, file in enumerate(files):
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

def get_ff_files(datadir):
    print("getting file paths")
    files = get_text_paths(datadir)
    print("parsing fanfiction")
    fflist = parse_fanfiction(files)
    return fflist



if __name__ == "__main__":
    path = "allbooks"
    books = get_ff_files(path)
    print(len(books))
    print(books[-1])

