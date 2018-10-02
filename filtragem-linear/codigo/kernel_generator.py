import random
from functools import reduce


def create_random_kernel(kernel_size):
    kernel = [float(random.randrange(0, 100)) for x in range(kernel_size)]
    media = reduce((lambda x, y: x + y), kernel) / kernel_size
    kernel = [x - media for x in kernel]
    norma = (reduce((lambda x, y: x + y * y), kernel)) ** 0.5
    kernel = [x / norma for x in kernel]
    return kernel + [0.0]


print("Enter kernel dimension (x y n_bands): ")
input_entry = input().split(' ')

if (len(input_entry) != 3):
    print('Parameters should x y n_bands')
    exit()

print("Enter number of kernels: ")
n_kernels = int(input())

x_size = int(input_entry[0])
y_size = int(input_entry[1])
n_bands = int(input_entry[2])
kernel_size = x_size * y_size * n_bands

kernel_bank = [create_random_kernel(kernel_size) for x in range(n_kernels)]

with open('random-kernal-output.txt', 'w') as output_file:
    output_file.writelines(str(n_bands) + ' ' + str(x_size) + ' ' +  str(y_size) + ' ' + str(n_kernels) + '\n')
    for kernel in kernel_bank:
        output_file.writelines(' '.join([str(x) for x in kernel]) + '\n')