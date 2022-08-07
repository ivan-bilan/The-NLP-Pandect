import re
import requests

with open("../README.md", encoding="utf8") as f:
    lines = f.readlines()

set_of_urls = set()

for i, line in enumerate(lines):
    urls = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', line)

    for url in urls:
        if "github.com" not in url and "png" not in url:
            set_of_urls.add(url)

response = None
for single_website in set_of_urls:
    try:
        response = requests.get(single_website)
    except Exception as e:
        print('Error reaching the website: ', e, single_website)
    else:
        if response.status_code != 200:
            print('Web site does not exist', single_website)
