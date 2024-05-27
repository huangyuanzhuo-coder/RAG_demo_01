import aspose.words as aw
import os

path = r'../bs_challenge_financial_14b_dataset/pdf'
file_name = []
path_list = []
for i in os.listdir(path):
    file_name.append(i)
    path_list.append(os.path.join(path + i))

for j in range(len(path_list)[:10]):
    doc = aw.Document(path_list[j])
    doc.save(r'../md_files/{}.md'.format(file_name[j].split('.')[0]))