#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: the number of processes must evenly divide %d.\n",
                    N);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // MPI_Bcast requires every process to hold a complete copy.
    int *array = malloc((size_t)N * sizeof(*array));
    if (array == NULL) {
        fprintf(stderr, "Rank %d: unable to allocate the array.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            array[i] = i + 1;
        }
        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    // This is collective: every rank must make the call.
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    int start_idx = rank * chunk_size;
    int end_idx = start_idx + chunk_size;
    long long local_sum = 0;

    for (int i = start_idx; i < end_idx; ++i) {
        local_sum += array[i];
    }

    printf("  Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, end_idx, local_sum);

    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;

        for (int source = 1; source < size; ++source) {
            long long received_sum;
            MPI_Recv(&received_sum, 1, MPI_LONG_LONG, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += received_sum;
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;

        printf("\n[Bcast] Total sum   = %lld\n", total_sum);
        printf("[Bcast] Expected    = %lld\n", expected);
        printf("[Bcast] Correct?    = %s\n",
               total_sum == expected ? "YES" : "NO");
        printf("[Bcast] Time        = %.4f sec\n", elapsed);
    }

    free(array);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
