import numpy as np
import shutil
import os

FILES = [os.path.join('submissions',i) for i in os.listdir('submissions')]

def read():
    res = []
    for file in FILES:
        _answers = {}
        with open(file, 'r') as f:
            temp = f.readlines()[1:]
        temp = [i.strip().split(',') for i in temp]
        _answers = {i[0]:int(i[1]) for i in temp}
        res.append(_answers)
    return res

def compare(answers):
    final = ['id,label']
    keys = answers[0].keys()
    count = 1
    for key in keys:
        if all([answers[0][key] == answers[i][key] for i in range(1,len(answers))]):
            final.append(f'{key},{answers[0][key]}')
        else:
            shutil.copy(f"data/test/{key}_{key}.wav",f"data/wtf/{key}.wav")
            print(key, ','.join([str(answers[i][key]) for i in range(len(answers))]))
            majority = int(sum([answers[i][key] for i in range(len(answers))]) > 0.5)
            count+=1
            #majority = int((answers[0][key] + answers[1][key] + answers[2][key])/3 > 0.5)
            final.append(f'{key},{majority}')
    print(count)
    with open('submission-ensemble.csv', 'w') as f:
        f.writelines('\n'.join(final))

def main():
    answers = read()
    compare(answers)
if __name__ == "__main__":
    main()
