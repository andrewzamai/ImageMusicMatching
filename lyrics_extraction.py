# if not working manage file != .DSstore
from nltk.corpus import stopwords
from string import punctuation
import os
from unidecode import unidecode

def contains_punctuation(word, list):
    for letter in word:
        if letter in list:
            return True
    return False


stopWords = set(stopwords.words('italian'))

lyrics_path = 'lyrics/'
lyrics_list = os.listdir('lyrics')

of = open('words_file.txt', 'w', encoding='utf-8')


for lyrics_file in lyrics_list:
    
    f = open(lyrics_path+lyrics_file, 'r', encoding='utf-8')
    lyric = f.read()
    lyric = unidecode(lyric)
    f.close()

    final_lyric = ''
    parentheses_open = ['(', '[', '{']
    parentheses_closed = [')', ']', '}']
    caps = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    parentheses_flag = False
    for i in range(len(lyric)): 
        tmp = lyric[i]
        if tmp in parentheses_open:
            parentheses_flag = True
            continue
        elif tmp in parentheses_closed: 
            parentheses_flag = False
            continue
        if parentheses_flag == False:
            if tmp == '\n': 
                final_lyric += ' '
            elif tmp in caps and lyric[i-1] != ' ':
                final_lyric += ' ' + tmp
            else:
                final_lyric += tmp

    tokens = final_lyric.split(' ')
    useful_tokens = []
    for word in tokens: 
        if word not in stopWords and len(word) > 3 and not contains_punctuation(word, punctuation): 
            useful_tokens.append(word.lower())
    words = lyrics_file[:-4] + ' '
    for word in useful_tokens:
        words += word + ','
    of.write(words[:-1]+'\n')

of.close()


    





