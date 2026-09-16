CC := mpicc
CFLAGS := -Wall -Wextra
MPIEXEC := mpirun
MPI_FLAGS := --oversubscribe
NP := 4

PROGRAMS := \
	Exercise01/sum_bcast \
	Exercise02/sum_scatter \
	Exercise03/sum_gather \
	Exercise04/sum_reduce \
	Exercise05/sum_allreduce \
	Exercise06/sum_scan

.PHONY: all run clean

all: $(PROGRAMS)

Exercise01/sum_bcast: Exercise01/sum_bcast.c
	$(CC) $(CFLAGS) -o $@ $<

Exercise02/sum_scatter: Exercise02/sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $<

Exercise03/sum_gather: Exercise03/sum_gather.c
	$(CC) $(CFLAGS) -o $@ $<

Exercise04/sum_reduce: Exercise04/sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $<

Exercise05/sum_allreduce: Exercise05/sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<

Exercise06/sum_scan: Exercise06/sum_scan.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	$(MPIEXEC) $(MPI_FLAGS) -np $(NP) ./Exercise01/sum_bcast
	$(MPIEXEC) $(MPI_FLAGS) -np $(NP) ./Exercise02/sum_scatter
	$(MPIEXEC) $(MPI_FLAGS) -np $(NP) ./Exercise03/sum_gather
	$(MPIEXEC) $(MPI_FLAGS) -np $(NP) ./Exercise04/sum_reduce
	$(MPIEXEC) $(MPI_FLAGS) -np $(NP) ./Exercise05/sum_allreduce
	$(MPIEXEC) $(MPI_FLAGS) -np $(NP) ./Exercise06/sum_scan

clean:
	$(RM) $(PROGRAMS)
