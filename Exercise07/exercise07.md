## Program Comparison

| Exercise | Program | Collective operations | Memory allocation | Manual summation on root | Final result availability |
|----------|---------|-----------------------|-------------------|---------------------------|---------------------------|
| 1 | sum_bcast | MPI_Bcast | Every process stores the complete array | Yes, root receives and adds each local sum | Root only |
| 2 | sum_scatter | MPI_Scatter | Root stores the complete array; other processes store only one chunk | Yes, root receives and adds each local sum | Root only |
| 3 | sum_gather | MPI_Scatter and MPI_Gather | Root stores the complete array and gathered sums; other processes store one chunk | Yes, root adds the gathered sums | Root only |
| 4 | sum_reduce | MPI_Scatter and MPI_Reduce | Root stores the complete array; other processes store only one chunk | No | Root only |
| 5 | sum_allreduce | MPI_Scatter and MPI_Allreduce | Root stores the complete array; other processes store only one chunk | No | All processes |
| 6 | sum_scan | MPI_Scatter and MPI_Scan | Root stores the complete array; other processes store only one chunk | No | A different prefix sum is available on each process; the last rank has the global total |

## Execution-Time Results

Record the time printed by each program after running it with 2, 4, and 8 processes.

| Program | 2 processes (seconds) | 4 processes (seconds) | 8 processes (seconds) |
|---------|-----------------------|-----------------------|-----------------------|
| sum_bcast | 0.0068 | 0.0289 | 0.0093 |
| sum_scatter | 0.0039 | 0.0037 | 0.0034 |
| sum_gather | 0.0031 | 0.0061 | 0.0043 |
| sum_reduce | 0.0089 | 0.0029 | 0.0049 |
| sum_allreduce | 0.0055 | 0.0177 | 0.0102 |
| sum_scan | 0.0131 | 0.0376 | 0.0107 |

## Performance Observation

The fastest program varied with the number of processes. MPI_Gather was fastest with 2 processes at 0.0031 seconds, MPI_Reduce was fastest with 4 processes at 0.0029 seconds, and MPI_Scatter was fastest with 8 processes at 0.0034 seconds. Across all three configurations, MPI_Scatter had the lowest average execution time at approximately 0.0037 seconds. The MPI_Scatter-based programs generally performed better than the MPI_Bcast version because each process received only the portion of the array that it needed instead of receiving all 1,000,000 elements. MPI_Reduce also performed well when only the root needed the final result because it combined the partial sums without requiring the root to receive or add each value manually. MPI_Allreduce required additional communication to make the final total available to every process, while MPI_Scan produced a separate cumulative result for every rank. All eighteen runs calculated the correct total of 500,000,500,000. The differences between individual measurements may also have been affected by system load, process startup overhead, available processor cores, and the MPI implementation.

## Thinking Question

MPI_Scan should be chosen over MPI_Allreduce when each process needs a cumulative value based on all processes up to its own rank instead of every process receiving the same global result. A concrete example is assigning global offsets to chunks of a distributed array. Each process first determines the number of items in its local chunk, and a prefix operation provides the total number of items assigned through that rank. By subtracting its local count, the process obtains the starting offset for its chunk in the global array. This allows all processes to calculate non-overlapping global positions without sending their counts to one process for manual processing.


