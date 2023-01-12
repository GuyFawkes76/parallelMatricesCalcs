# coding=utf-8
from numba import cuda, float32
import numpy as np
import math
import time
import sys

@cuda.jit
def cuda_multiplication(a, b, c):
    # Задаёт массив в общей памяти
    # Размер и тип массивов должны быть известны к моменту компиляции
    s_a = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    s_b = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    # Получает абсолютную позицию текущего потока в сетке блоков
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # блоков на сетку

    # Каждый поток считает один элемент в результирующей матрице.
    # Дробное произведение квантуется на дробные произведения векторов длиной TBP.
    tmp = float32(0.)
    for i in range(bpg):
        # Предзагружает данные в общую память
        s_a[ty, tx] = 0
        s_b[ty, tx] = 0
        if y < a.shape[0] and (tx + i * TPB) < a.shape[1]:
          s_a[ty, tx] = a[y, tx + i * TPB]
        if x < b.shape[1] and (ty + i * TPB) < b.shape[0]:
          s_b[ty, tx] = b[ty + i * TPB, x]

        # Ждёт пока все потоки завершат предзагрузку
        cuda.syncthreads()

        # Считает частичное произведение в общей памяти
        for j in range(TPB):
            tmp += s_a[ty, j] * s_b[j, tx]

        # Ждёт пока все потоки завершат вычисления
        cuda.syncthreads()
    if y < c.shape[0] and x < c.shape[1]:
        c[y, x] = tmp


def main():
    dim_x_1 = dim_z_1 = 4096
    dim_x_2 = dim_y_1 = 2048
    dim_y_2 = dim_z_2 = 8192
    # Задаёт количество потоков на блок, и управляет использованием общей памяти.
    # Расчёты ведутся на блоках из TPBxTPB элементов.
    # TPB должно быть меньше 32.
    x_h = np.arange(dim_x_1 * dim_x_2).reshape([dim_x_1, dim_x_2])
    y_h = np.ones([dim_y_1, dim_y_2])
    z_h = np.zeros([dim_z_1, dim_z_2])

    x_d = cuda.to_device(x_h)
    y_d = cuda.to_device(y_h)
    z_d = cuda.to_device(z_h)

    threadsperblock = (TPB, TPB)
    grid_y_max = max(x_h.shape[0], y_h.shape[0])
    grid_x_max = max(x_h.shape[1], y_h.shape[1])
    blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0])
    blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()
    cuda_multiplication[blockspergrid, threadsperblock](x_d, y_d, z_d)
    z_h = z_d.copy_to_host()
    end_time = time.time()
    # print(z_h)
    with open('results.csv', mode='a+') as f:
        f.write("{},{}\n".format(TPB, end_time - start_time))
    # print(x_h@y_h)

TPB = int(sys.argv[1])
main()
