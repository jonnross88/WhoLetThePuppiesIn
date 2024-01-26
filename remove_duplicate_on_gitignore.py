filename = ".gitignore"
with open(filename) as f:
    content = f.readlines()
content = list(dict.fromkeys(content))
with open(filename, "w") as f:
    f.writelines(content)
