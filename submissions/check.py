import numpy as np

def read(filename):
    with open(filename, 'r') as f:
        data = f.readlines()[1:]
    data = [da.strip().split(',') for da in data]
    data = {da[0]:int(da[1]) for da in data}
    return data
def main():
    best = read('./9775.csv')
    human = read('./labels.csv')
    for key in best.keys():
        if key not in human:
            continue
        if best[key] != human[key]:
            best[key] = human[key]
            print(key)
    res = ['id,label']
    for i,j in best.items():
        res.append(f'{i},{j}')
    #with open('submission-f.csv','w') as f:
    #    f.writelines('\n'.join(res))
if __name__ == "__main__":
    main()
