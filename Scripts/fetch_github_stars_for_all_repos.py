"""
This script will parse all GitHub repositories present in the README.md, update the number of stars for each, and
regenerate the README.md with the updated star count. All you need is a GitHub API token that you can generate in your
GitHub profile settings.
"""


import re
import requests

from tqdm import tqdm

from github_access_token import github_access_token

with open("../README.md", encoding="utf8") as f:
    lines = f.readlines()

for i, line in tqdm(enumerate(lines)):
    if "github.com" in line:
        try:
            username, project_name = re.search(r".*?github.com/(.*?)/(.*?)\).*?", line).groups()
            project_name = project_name.split("/")[0]

            query_url = f"https://api.github.com/repos/{username}/{project_name}/".strip(
                "/"
            )
            params = {
                "state": "open",
            }
            headers = {'Authorization': f'token {github_access_token}'}
            r = requests.get(query_url, headers=headers, params=params)
            # print(r.json())
            number_of_stars = r.json()["stargazers_count"]
            line = re.sub(
                r"(?is)GitHub,? .*? ?stars",
                f"GitHub, {number_of_stars} stars",
                line,
            )
            lines[i] = line
        except (KeyError, AttributeError) as e:
            print(line, e)
            pass

with open('README.md', 'a', encoding="utf8") as f:
    for item in lines:
        f.write(item)
