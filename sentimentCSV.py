import os
import subprocess
import sys
import timeit
import csv
import pandas as pd
import re
import string
import numpy as np
import shutil

directory = os.getcwd()

os.chdir(directory)

input_filename = raw_input("Enter input filename: ")
output_filename = raw_input("Enter output filename: ")

text_column = raw_input("Enter text column name: ")

print "Reading file..."
data = pd.read_csv(input_filename, error_bad_lines=False)
print "Cleaning comments..."
comments = data[text_column]
comments_clean = []

for comment in comments:
    comment = re.sub(r'\n', r'',str(comment))
    comment = re.sub(r'MR', r'',str(comment))
    comment = re.sub(r'mr', r'',str(comment))
    comment = re.sub(r'Mr', r'',str(comment))
    comment = ' '.join(re.split(r'(?<=[.:;])\s', comment)[:1])
    comment  = comment.translate(string.maketrans("",""), string.punctuation)
    comments_clean.append(comment)
    
comment_chunks=[comments_clean[x:x+2000] for x in xrange(0, len(comments_clean), 2000)]

input_directory = directory + '/sentiment-engine/input_data'
if not os.path.exists(input_directory):
    os.makedirs(input_directory)
os.chdir(input_directory)

N = len(comment_chunks)
for n in range(N):
    f = open("comments" + str(n) + ".txt", "w");
    comments = comment_chunks[n]
    for i in range(len(comments)):
        if i == len(comments)-1:
            f.write(str(comments[i]))
            f.write(".")
        else:
            f.write(str(comments[i]))
            f.write(". \n")
    f.close() 
    
os.chdir(directory + '/sentiment-engine/')

sentiments = ['  Neutral', '  Negative', '  Positive', '  Very positive', '  Very negative']

def chunks(l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in xrange(0, len(l), n):
            yield l[i:i+n]

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "-"*block + " "*(barLength-block), round(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()

f = open("output.csv", "wb")

print "Calculating Sentiment..."

start = timeit.default_timer()
    
for n in range(N):
    
    file_name = os.path.join('input_data', 'comments' + str(n) + '.txt')
    

    p = subprocess.Popen('java -cp "*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file ' + file_name,
              shell=True,
              stdout=subprocess.PIPE)
    output, errors = p.communicate()
    
    senti_list = output.split('\n')
    
    del output, errors

    for i in range(len(senti_list)):
        if i % 2 == 1 and senti_list[i] not in sentiments:
            senti_list.insert(i, '  Neutral')

    senti_list = senti_list[:-1]
    
    output_list = list(chunks(senti_list, 2))
    
    progress = float(n)/N
    update_progress(progress)
    #print "rows:", len(output_list)
    
    
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(output_list)
    
    del senti_list, output_list
    
f.close()

shutil.rmtree(directory + '/sentiment-engine/input_data/')

stop = timeit.default_timer()
print "Time taken:", stop - start

output_frame = pd.read_csv("output.csv", header=None)

output_frame.columns = ['Text', 'Sentiment']
senti_text = np.array(output_frame['Text'])
senti_bool = []
for element in senti_text:
    if element == '.':
        senti_bool.append(0)
    else:
        senti_bool.append(1)
output_frame["Text_Bool"] =  pd.Series(senti_bool)
del senti_bool

data['Sentiment'] = output_frame['Sentiment']

data['Text_Bool'] = output_frame['Text_Bool']

os.chdir('..')
print "Writing to output file..."
data.to_csv(output_filename)#, encoding = 'utf-8-sig')
print "Finished!"
