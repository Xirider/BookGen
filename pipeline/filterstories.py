from joblib import Memory

cachedir = "cache"
memory = Memory(cachedir, verbose=10)



# @memory.cache
def filter_ff_stories(books, max_rating, min_words, max_words, min_chapters, max_chapters, max_books):
    print("filtering ff stories")
    ratings = {"K":1, "K+":2, "T":3, "M":4, "MA":5 }
    rating_number = ratings[max_rating]
    delete_ids = []
    for bookid, book in enumerate(books):
        if bookid % 1000 == 0:
            print(f"filtering book {bookid} now")
        removal = False

        if book["Language"] != "English":
            removal = True
        
        if ratings[book["Rating"]] > rating_number:
            removal = True

        words = int(book["Words"].replace(",",""))
        if not (min_words <= words <= max_words):
            removal = True

        chapters = int(book["Chapters"].replace(",",""))
        if not (min_chapters <= chapters <= max_chapters):
            removal = True


        
        if removal:
            delete_ids.append(bookid)

    for bookid in reversed(delete_ids):
        del books[bookid]

    books = books[:max_books]   

     
    return books