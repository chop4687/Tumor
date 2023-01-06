import matplotlib.pyplot as plt

if __name__ == '__main__':
    name = ['AB','OKC','DC','NORMAL']
    recall = [0.8238,0.4444,0.7488,0.3614]
    precision = [0.8650,0.2666,0.8808,0.3092]
    plt.plot(precision[0], recall[0],label='AB')
    plt.savefig('sdf.png')
