from mpi4py import MPI
import numpy as np
import time

# Константы
NX, NY, NZ = 450, 450, 450
NUM_STEPS = 1000
DT = 0.01
ALPHA = 0.00001172
DX, DY, DZ = 0.001, 0.001, 0.001
OUTPUT_INTERVAL = 100

CUBE_SIZE = 90
INITIAL_TEMP = 1000.0
BOUNDARY_TEMP = 20.0

def solve_heat_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Распределение по Z
    base_nz = NZ // size
    start_z = rank * base_nz
    end_z = NZ if rank == size - 1 else start_z + base_nz
    local_nz = end_z - start_z
    local_nz_with_halo = local_nz + 2

    if rank == 0:
        print(f"MPI Python FIXED: {size} processes")
        print(f"Grid: {NX}x{NY}x{NZ}, Steps: {NUM_STEPS}")

    # Выделение памяти в порядке [k][j][i]
    local_temp = np.empty((local_nz_with_halo, NY, NX), dtype=np.float64)
    local_new  = np.empty_like(local_temp)

    # Инициализируем все точки boundary temperature
    local_temp[:] = BOUNDARY_TEMP
    local_new[:]  = BOUNDARY_TEMP

    # Инициализация горячего куба
    center_x = NX // 2
    center_y = NY // 2
    center_z = NZ // 2

    i = np.arange(NX)[None, None, :]
    j = np.arange(NY)[None, :, None]
    k_global = np.arange(start_z, end_z)[:, None, None]

    hot_mask = (
        (np.abs(i - center_x) < CUBE_SIZE) &
        (np.abs(j - center_y) < CUBE_SIZE) &
        (np.abs(k_global - center_z) < CUBE_SIZE)
    )

    # Помещаем горячий блок в local_temp[1:local_nz+1]
    local_temp[1:local_nz+1] = np.where(
        hot_mask, INITIAL_TEMP, local_temp[1:local_nz+1]
    )
    local_new[1:local_nz+1] = local_temp[1:local_nz+1]

    # Коэффициенты
    dx2 = DX * DX
    dy2 = DY * DY
    dz2 = DZ * DZ
    factor = DT * ALPHA

    start_time = time.time()

    # Основной цикл
    for step in range(NUM_STEPS):

        # Обмен halo-областями
        if size > 1:

            # Отправляем нижний слой предыдущему процессу, получаем верхний halo
            if rank > 0:
                sendbuf = np.ascontiguousarray(local_temp[1, :, :])
                recvbuf = np.full_like(sendbuf, BOUNDARY_TEMP)
                comm.Sendrecv(
                    sendbuf=sendbuf, dest=rank - 1, sendtag=0,
                    recvbuf=recvbuf, source=rank - 1, recvtag=1
                )
                local_temp[0, :, :] = recvbuf
            else:
                local_temp[0, :, :] = BOUNDARY_TEMP

            # Отправляем верхний слой следующему процессу, получаем нижний halo
            if rank < size - 1:
                sendbuf2 = np.ascontiguousarray(local_temp[local_nz, :, :])
                recvbuf2 = np.full_like(sendbuf2, BOUNDARY_TEMP)
                comm.Sendrecv(
                    sendbuf=sendbuf2, dest=rank + 1, sendtag=1,
                    recvbuf=recvbuf2, source=rank + 1, recvtag=0
                )
                local_temp[local_nz + 1, :, :] = recvbuf2
            else:
                local_temp[local_nz + 1, :, :] = BOUNDARY_TEMP

        else:
            local_temp[0, :, :] = BOUNDARY_TEMP
            local_temp[-1, :, :] = BOUNDARY_TEMP

        # Вычисления - только внутренние точки
        core = local_temp[1:-1, 1:-1, 1:-1]

        d2x = (local_temp[1:-1, 1:-1, 2:] -
               2 * core +
               local_temp[1:-1, 1:-1, :-2]) / dx2

        d2y = (local_temp[1:-1, 2:, 1:-1] -
               2 * core +
               local_temp[1:-1, :-2, 1:-1]) / dy2

        d2z = (local_temp[2:, 1:-1, 1:-1] -
               2 * core +
               local_temp[:-2, 1:-1, 1:-1]) / dz2

        local_new[1:-1, 1:-1, 1:-1] = core + factor * (d2x + d2y + d2z)
        local_temp[1:-1, 1:-1, 1:-1] = local_new[1:-1, 1:-1, 1:-1]

        # Граничные условия Дирихле на глобальных границах X и Y
        local_temp[:, :, 0]      = BOUNDARY_TEMP
        local_temp[:, :, NX - 1] = BOUNDARY_TEMP
        local_temp[:, 0, :]      = BOUNDARY_TEMP
        local_temp[:, NY - 1, :] = BOUNDARY_TEMP

        # Глобальные граничные условия по Z для крайних процессов
        if rank == 0:
            local_temp[0, :, :] = BOUNDARY_TEMP
        if rank == size - 1:
            local_temp[local_nz + 1, :, :] = BOUNDARY_TEMP

        # Вывод прогресса
        if rank == 0 and step % OUTPUT_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{NUM_STEPS}, Time: {elapsed:.2f}s")

    total_time = time.time() - start_time

    # Сохранение результатов
    if rank == 0:

        global_grid = np.empty((NX, NY, NZ), dtype=np.float64)

        # Копируем данные процесса 0
        global_grid[:, :, :local_nz] = local_temp[1:local_nz+1].transpose(2, 1, 0)

        # Получаем данные от других процессов
        for proc in range(1, size):
            proc_base_nz = NZ // size
            proc_start = proc * proc_base_nz
            proc_nz = NZ - proc_start if proc == size - 1 else proc_base_nz

            buf = np.empty((proc_nz, NY, NX), dtype=np.float64)
            comm.Recv(buf, source=proc, tag=0)

            global_grid[:, :, proc_start:proc_start+proc_nz] = buf.transpose(2, 1, 0)

        # Сохраняем 3D-сетку в бинарный файл
        with open("python_mpi_results.bin", "wb") as f:
            for i in range(NX):
                for j in range(NY):
                    global_grid[i, j, :].tofile(f)

        print(f"MPI Python ({size} processes): {total_time:.2f} sec")
        print("Saved to python_mpi_results.bin")

    else:
        # Отправляем данные процессу 0 (без halo)
        sendbuf = np.ascontiguousarray(local_temp[1:local_nz + 1])
        comm.Send(sendbuf, dest=0, tag=0)


if __name__ == "__main__":
    solve_heat_mpi()