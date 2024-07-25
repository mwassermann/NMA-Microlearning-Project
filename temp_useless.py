import matplotlib.pyplot as plt
import numpy as np


# 读取 .npy 文件的函数
def read_npy_file(file_path):
    return np.load(file_path)

# 画图的函数
def plot_data(data, title, ax, X, Y):
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel(X)
    ax.set_ylabel(Y)

def main():
    # 文件路径
    file1 = 'results/netffa/losses_ffa_nonstat_ce.npy'
    file2 = 'results/netffa/test_loss_ffa_nonstat_ce.npy'
    file3 = 'results/netffa/losses_ffa_nonstat_mse.npy'
    file4 = 'results/netffa/test_loss_ffa_nonstat_mse.npy'
    file5 = 'results/netffa/accuracy_ffa_nonstat.npy'
    
    
    # 读取文件
    data1 = read_npy_file(file1)
    data2 = read_npy_file(file2)
    data3 = read_npy_file(file3)
    data4 = read_npy_file(file4)
    data5 = read_npy_file(file5)
    
    # 输出
    print(data1.shape)
    print(data2.shape)
    print(data3.shape)
    print(data4.shape)
    print(data5.shape)
    

    # 创建子图
    fig, axes = plt.subplots(5, 1, figsize=(10, 8))
    
    fig.suptitle('FFA nonstat')

    # 在子图中绘制数据
    plot_data(data1, 'Training Loss', axes[0], "Batches", "CE")
    plot_data(data2, 'Test Loss', axes[1], "Epochs", "CE")
    plot_data(data3, 'Training Loss', axes[2], "Batches", "MSE")
    plot_data(data4, 'Test Loss', axes[3], "Epochs", "MSE")
    plot_data(data5, 'Accuracy', axes[4], "Epochs", "Accuracy")

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
