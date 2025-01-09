#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank;
    long int n;

    // Инициализация MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Проверка аргументов командной строки
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Использование: %s <n>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Преобразование аргумента в число
    n = strtol(argv[1], NULL, 10);

    // Процесс с рангом 0
    if (rank == 0) {
        int *a = (int *)malloc(n * sizeof(int));
        for (long int i = 0; i < n; i++) {
            a[i] = 0; // Заполнение массива нулями
        }

        double start_time = MPI_Wtime(); // Начало таймера

        // Отправка массива на процесс с рангом 1
        MPI_Send(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Получение массива обратно
        MPI_Recv(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Отправка массива на процесс с рангом 1
        MPI_Send(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Получение массива обратно
        MPI_Recv(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double end_time = MPI_Wtime(); // Конец таймера

        double elapsed_time = (end_time - start_time) / 4.0; // Делим пополам

        // Запись времени в файл
        FILE *file = fopen("my_results.txt", "a");
        if (file != NULL) {
            fprintf(file, "Time %f size %ld\n", elapsed_time, n);
            fclose(file);
        } else {
            perror("Не удалось открыть results.txt");
        }

        free(a);
    }

    // Процесс с рангом 1
    else if (rank == 1) {
        int *a = (int *)malloc(n * sizeof(int));
        
        // Получение массива от процесса с рангом 0
        MPI_Recv(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Отправка массива обратно на процесс с рангом 0
        MPI_Send(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        // Получение массива от процесса с рангом 0
        MPI_Recv(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Отправка массива обратно на процесс с рангом 0
        MPI_Send(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        

        
        free(a);
    }

    // Завершение работы MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
