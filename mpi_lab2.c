#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank;
    long int n;

    // Èíèöèàëèçàöèÿ MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Ïðîâåðêà àðãóìåíòîâ êîìàíäíîé ñòðîêè
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Èñïîëüçîâàíèå: %s <n>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Ïðåîáðàçîâàíèå àðãóìåíòà â ÷èñëî
    n = strtol(argv[1], NULL, 10);

    // Ïðîöåññ ñ ðàíãîì 0
    if (rank == 0) {
        int *a = (int *)malloc(n * sizeof(int));
        for (long int i = 0; i < n; i++) {
            a[i] = 0; // Çàïîëíåíèå ìàññèâà íóëÿìè
        }

        double start_time = MPI_Wtime(); // Íà÷àëî òàéìåðà

        // Îòïðàâêà ìàññèâà íà ïðîöåññ ñ ðàíãîì 1
        MPI_Send(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Ïîëó÷åíèå ìàññèâà îáðàòíî
        MPI_Recv(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Îòïðàâêà ìàññèâà íà ïðîöåññ ñ ðàíãîì 1
        MPI_Send(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Ïîëó÷åíèå ìàññèâà îáðàòíî
        MPI_Recv(a, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double end_time = MPI_Wtime(); // Êîíåö òàéìåðà

        double elapsed_time = (end_time - start_time) / 4.0; // Äåëèì ïîïîëàì

        // Çàïèñü âðåìåíè â ôàéë
        FILE *file = fopen("my_results.txt", "a");
        if (file != NULL) {
            fprintf(file, "Time %f size %ld\n", elapsed_time, n);
            fclose(file);
        } else {
            perror("Íå óäàëîñü îòêðûòü results.txt");
        }

        free(a);
    }

    // Ïðîöåññ ñ ðàíãîì 1
    else if (rank == 1) {
        int *a = (int *)malloc(n * sizeof(int));
        
        // Ïîëó÷åíèå ìàññèâà îò ïðîöåññà ñ ðàíãîì 0
        MPI_Recv(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Îòïðàâêà ìàññèâà îáðàòíî íà ïðîöåññ ñ ðàíãîì 0
        MPI_Send(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        // Ïîëó÷åíèå ìàññèâà îò ïðîöåññà ñ ðàíãîì 0
        MPI_Recv(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Îòïðàâêà ìàññèâà îáðàòíî íà ïðîöåññ ñ ðàíãîì 0
        MPI_Send(a, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        

        
        free(a);
    }

    // Çàâåðøåíèå ðàáîòû MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
