import os

kl = os.listdir('collected_images')
print(kl)
print(len(kl))

with open('labels.txt', 'w') as oiu:
    labels_string = '\n'.join(kl)  # Convert the list to a string with each label on a new line
    oiu.write(labels_string)
