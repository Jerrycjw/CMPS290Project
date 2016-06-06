import uuid
from newspaper import *
import os
f1 = open('urls_1.txt','r+')
f2 = open('urls_2.txt','r+')
temp1 = f1.read().splitlines()
temp2 = f2.read().splitlines()
url_filtered = filter(lambda x: x not in temp1, temp2)
i = 500
for url in url_filtered:
	i=i+1
	with open("dataset/record"+str(i)+".tsv", "w+") as record_file:
		try:
			x = Article(url, language='en')
			x.download()
			x.parse()
			record_file.write(str(x.text)+"\n")
		except ArticleException:
			pass
