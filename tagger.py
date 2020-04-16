#used for converting sentence into pos tag
from nltk import word_tokenize, pos_tag

#it helps to convert pos tags into vectors
posVectors = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
filters = ["MD","VBP","VBZ","VBG","VBD","VBN","VB"]

#function to convert sentense to pos tags
def tag(sentense):
    try:
        vectors = []
        filterPosList = []
        posList = pos_tag(word_tokenize(sentense))
        print(posList)
        for data in posList:
            filterPosList.append(data[1])
        filterPosList = list(set(filterPosList) & set(filters))
        print(filterPosList)
        for tags in filters:
            if tags in filterPosList:
                vectors.append(1)
            else:
                vectors.append(0)
        return (vectors)
    except:
        return False

#print(tag("have to change engine oil"))
