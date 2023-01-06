import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.title('Accuracy')
    plt.bar(['Normal','AB','DC','OKC'],[0.855, 0.2533, 0.715, 0.4227])
    plt.savefig('Accuracy_without.png')
