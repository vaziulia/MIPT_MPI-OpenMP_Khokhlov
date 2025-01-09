#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank;
    long int n;

    // ������������� MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // �������� ���������� ��������� ������
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "�������������: %s <n>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // �������������� ��������� � �����
    n = strtol(argv[1], NULL, 10);

    // ������� � ������ 0
    if (rank == 0) {
        int *a = (int *)malloc(n * sizeof(int));
        for (long int i = 0; i < n; i++) {
            a[i] = 0; // ���������� ������� ������
        }

        double start_time = MPI_Wtime(); // ������ �������

        // �������� ������� �� ������� � ������ 1
        MPI_Send(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // ��������� ������� �������
        MPI_Recv(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // �������� ������� �� ������� � ������ 1
        MPI_Send(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // ��������� ������� �������
        MPI_Recv(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double end_time = MPI_Wtime(); // ����� �������

        double elapsed_time = (end_time - start_time) / 4.0; // ����� �������

        // ������ ������� � ����
        FILE *file = fopen("my_results.txt", "a");
        if (file != NULL) {
            fprintf(file, "Time %f size %ld\n", elapsed_time, n);
            fclose(file);
        } else {
            perror("�� ������� ������� results.txt");
        }

        free(a);
    }

    // ������� � ������ 1
    else if (rank == 1) {
        int *a = (int *)malloc(n * sizeof(int));
        
        // ��������� ������� �� �������� � ������ 0
        MPI_Recv(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // �������� ������� ������� �� ������� � ������ 0
        MPI_Send(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        // ��������� ������� �� �������� � ������ 0
        MPI_Recv(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // �������� ������� ������� �� ������� � ������ 0
        MPI_Send(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        

        
        free(a);
    }

    // ���������� ������ MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
