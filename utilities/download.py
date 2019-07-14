import urllib.request
import sys
import os

txtfile = open(sys.argv[1], "r")
urls = txtfile.readlines()
os.mkdir('images')

# Add itr in case the files have similar names
for itr, url in enumerate(urls):
    urlseparated = url.split('/')
    urlpostfix = urlseparated[-1]
    urllib.request.urlretrieve(url, "images/" + str(itr) + urlpostfix)