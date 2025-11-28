#include "common.h"
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Распределение по Z
    int base_nz = NZ / size;
    int start_z = rank * base_nz;
    int end_z = (rank == size - 1) ? NZ : start_z + base_nz;
    int local_nz = end_z - start_z;
    int local_nz_with_halo = local_nz + 2;

    if (rank == 0) {
        printf("MPI C version: %d processes\n", size);
        printf("Global grid: %dx%dx%d, Steps: %d\n", NX, NY, NZ, NUM_STEPS);
    }

    // Выделение памяти в порядке [k][j][i]
    double* temp = malloc(sizeof(double) * local_nz_with_halo * NY * NX);
    double* new_temp = malloc(sizeof(double) * local_nz_with_halo * NY * NX);

    if (temp == NULL || new_temp == NULL) {
        printf("Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Инициализируем все точки boundary temperature (20.0)
    for (int k = 0; k < local_nz_with_halo; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = k * NY * NX + j * NX + i;
                temp[idx] = BOUNDARY_TEMP;
                new_temp[idx] = BOUNDARY_TEMP;
            }
        }
    }

    // Инициализация горячего куба (без halo)
    int center_x = NX/2, center_y = NY/2, center_z = NZ/2;
    for (int k_local = 1; k_local <= local_nz; k_local++) {
        int k_global = start_z + (k_local - 1);
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if (abs(i - center_x) < CUBE_SIZE && 
                    abs(j - center_y) < CUBE_SIZE && 
                    abs(k_global - center_z) < CUBE_SIZE) {
                    int idx = k_local * NY * NX + j * NX + i;
                    temp[idx] = INITIAL_TEMP;
                    new_temp[idx] = INITIAL_TEMP;
                }
            }
        }
    }

    // Коэффициенты
    double dx2 = DX * DX;
    double dy2 = DY * DY;
    double dz2 = DZ * DZ;
    double factor = DT * ALPHA;

    double start_time = MPI_Wtime();

    for (int step = 0; step < NUM_STEPS; step++) {
        // Обмен halo-областями
        if (size > 1) {
            MPI_Request requests[4];
            int request_count = 0;
            
            // Отправляем нижний слой предыдущему процессу, получаем верхний halo
            if (rank > 0) {
                MPI_Isend(&temp[1 * NY * NX], NY * NX, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Irecv(&temp[0 * NY * NX], NY * NX, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
            }
            
            // Отправляем верхний слой следующему процессу, получаем нижний halo
            if (rank < size - 1) {
                MPI_Isend(&temp[local_nz * NY * NX], NY * NX, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Irecv(&temp[(local_nz + 1) * NY * NX], NY * NX, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
            }
            
            // Ждем завершения всех операций
            if (request_count > 0) {
                MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            }
        }

        // Вычисления - только внутренние точки
        for (int k_local = 1; k_local <= local_nz; k_local++) {
            for (int j = 1; j < NY-1; j++) {
                for (int i = 1; i < NX-1; i++) {
                    int idx = k_local * NY * NX + j * NX + i;
                    int idx_xp = k_local * NY * NX + j * NX + (i+1);
                    int idx_xm = k_local * NY * NX + j * NX + (i-1);
                    int idx_yp = k_local * NY * NX + (j+1) * NX + i;
                    int idx_ym = k_local * NY * NX + (j-1) * NX + i;
                    int idx_zp = (k_local+1) * NY * NX + j * NX + i;
                    int idx_zm = (k_local-1) * NY * NX + j * NX + i;

                    double d2x = (temp[idx_xp] - 2*temp[idx] + temp[idx_xm]) / dx2;
                    double d2y = (temp[idx_yp] - 2*temp[idx] + temp[idx_ym]) / dy2;
                    double d2z = (temp[idx_zp] - 2*temp[idx] + temp[idx_zm]) / dz2;
                    
                    new_temp[idx] = temp[idx] + factor * (d2x + d2y + d2z);
                }
            }
        }

        // Копирование обновленных значений
        for (int k_local = 1; k_local <= local_nz; k_local++) {
            for (int j = 1; j < NY-1; j++) {
                for (int i = 1; i < NX-1; i++) {
                    int idx = k_local * NY * NX + j * NX + i;
                    temp[idx] = new_temp[idx];
                }
            }
        }

        // Граничные условия Дирихле на глобальных границах X и Y
        for (int k = 0; k < local_nz_with_halo; k++) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    int idx = k * NY * NX + j * NX + i;
                    // Границы по X
                    if (i == 0 || i == NX-1) {
                        temp[idx] = BOUNDARY_TEMP;
                        new_temp[idx] = BOUNDARY_TEMP;
                    }
                    // Границы по Y
                    if (j == 0 || j == NY-1) {
                        temp[idx] = BOUNDARY_TEMP;
                        new_temp[idx] = BOUNDARY_TEMP;
                    }
                }
            }
        }

        // Глобальные граничные условия по Z для крайних процессов
        if (rank == 0) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    int idx = 0 * NY * NX + j * NX + i;
                    temp[idx] = BOUNDARY_TEMP;
                    new_temp[idx] = BOUNDARY_TEMP;
                }
            }
        }
        if (rank == size - 1) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    int idx = (local_nz + 1) * NY * NX + j * NX + i;
                    temp[idx] = BOUNDARY_TEMP;
                    new_temp[idx] = BOUNDARY_TEMP;
                }
            }
        }

        // Вывод прогресса
        if (rank == 0 && step % OUTPUT_INTERVAL == 0) {
            double elapsed = MPI_Wtime() - start_time;
            printf("Step %d/%d, Time: %.2fs\n", step, NUM_STEPS, elapsed);
        }
    }

    double total_time = MPI_Wtime() - start_time;

    // Сохранение результатов
    if (rank == 0) {
        FILE* file = fopen("mpi_results.bin", "wb");
        if (!file) {
            printf("Error opening file for writing\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Создаем глобальную сетку в порядке [i][j][k]
        double* global_grid = malloc(NX * NY * NZ * sizeof(double));
        
        // Копируем данные процесса 0
        for (int k_local = 0; k_local < local_nz; k_local++) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    int src_idx = (k_local + 1) * NY * NX + j * NX + i;
                    int dst_idx = i * NY * NZ + j * NZ + (start_z + k_local);
                    global_grid[dst_idx] = temp[src_idx];
                }
            }
        }

        // Получаем данные от других процессов
        for (int proc = 1; proc < size; proc++) {
            int proc_base_nz = NZ / size;
            int proc_start = proc * proc_base_nz;
            int proc_nz = (proc == size - 1) ? (NZ - proc_start) : proc_base_nz;
            
            double* proc_data = malloc(proc_nz * NY * NX * sizeof(double));
            MPI_Recv(proc_data, proc_nz * NY * NX, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int k_local = 0; k_local < proc_nz; k_local++) {
                for (int j = 0; j < NY; j++) {
                    for (int i = 0; i < NX; i++) {
                        int src_idx = k_local * NY * NX + j * NX + i;
                        int dst_idx = i * NY * NZ + j * NZ + (proc_start + k_local);
                        global_grid[dst_idx] = proc_data[src_idx];
                    }
                }
            }
            free(proc_data);
        }

        // Сохраняем 3D-сетку в бинарный файл
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                fwrite(&global_grid[i * NY * NZ + j * NZ], sizeof(double), NZ, file);
            }
        }
        
        fclose(file);
        free(global_grid);
        printf("Results saved to mpi_results.bin\n");
        
    } else {
        // Отправляем данные процессу 0 (без halo)
        double* send_data = malloc(local_nz * NY * NX * sizeof(double));
        
        for (int k_local = 0; k_local < local_nz; k_local++) {
            for (int j = 0; j < NY; j++) {
                for (int i = 0; i < NX; i++) {
                    int src_idx = (k_local + 1) * NY * NX + j * NX + i;
                    int dst_idx = k_local * NY * NX + j * NX + i;
                    send_data[dst_idx] = temp[src_idx];
                }
            }
        }
        
        MPI_Send(send_data, local_nz * NY * NX, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        free(send_data);
    }

    if (rank == 0) {
        printf("MPI C (%d processes): %.2f sec\n", size, total_time);
    }

    free(temp);
    free(new_temp);
    MPI_Finalize();
    return 0;
}