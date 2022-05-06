def getFilteredMusicList(detectedObjectsList, pathToFileWithAllLyrics):

    pairs_dict = {}
    with open(pathToFileWithAllLyrics, 'r') as words_file:
        while (True):
            song = words_file.readline()
            if not song: 
                break
            data = song.split(' ')
            name = data[0]
            words = data[1]
            words = words.split(',')
            for word in words: 
                if word in detectedObjectsList:
                    if name+'>'+word not in pairs_dict:
                        pairs_dict[name+'>'+word] = 1
                    else: 
                        pairs_dict[name+'>'+word] += 1
        #words_file.close()
    
    max = [('', 0), ('', 0), ('', 0), ('', 0), ('', 0), ('', 0), ('', 0), ]
    max_mod = True
    for key in pairs_dict.keys():
        min_index = -1
        if max_mod == True:
            min = 1000
            for i in range(0, len(max)):
                if max[i][1] < min: 
                    min = max[i][1]
                    min_index = i
        
        if min_index != -1:
            if pairs_dict[key] > max[min_index][1]:
                max_mod = True
                max[min_index] = (key, pairs_dict[key])
        else: 
            max_mod = False
    
    max = sorted(max, reverse=True, key=lambda x : x[1])
    
    return max


