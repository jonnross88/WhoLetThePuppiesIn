# python file to sort the data in requireents.txt file
with open("requirements.txt", "r") as f:
    data = f.readlines()
    data.sort()
# write data to file again
with open("requirements.txt", "w") as f:
    f.writelines(data)
