import os

img_root = '../data/traffic/images/'

def generate_dataset_txt(img_root):
    li = sorted(os.listdir(img_root))

    if os.path.exists('../dataset.txt'):
        os.remove('../dataset.txt')
    with open('../dataset.txt', 'a+') as f:
        for i in range(len(li)):
            img = li[i]
            img_path = img_root[3:] + img + '\n'
            f.write(img_path)

if __name__ == "__main__":

    generate_dataset_txt(img_root)