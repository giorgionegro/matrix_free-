#define _GNU_SOURCE
#include <mpi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct
{
  double collective_time;
  double p2p_time;
  double wait_time;
  double p2p_wait_time;
  double total_runtime;

  uint64_t collective_calls;
  uint64_t p2p_calls;
  uint64_t wait_calls;

  uint64_t collective_bytes;
  uint64_t p2p_send_bytes;
  uint64_t p2p_recv_bytes;

  // Per-function counters for newly added wrappers
  uint64_t calls_allgatherv;
  uint64_t calls_alltoallv;
  uint64_t calls_cancel;
  uint64_t calls_exscan;
  uint64_t calls_gatherv;
  uint64_t calls_iallreduce;
  uint64_t calls_ibarrier;
  uint64_t calls_issend;
  uint64_t calls_reduce_scatter;
  uint64_t calls_reduce_scatter_block;
  uint64_t calls_request_free;
  uint64_t calls_rsend;
  uint64_t calls_scan;
  uint64_t calls_scatterv;
  uint64_t calls_ssend;
  uint64_t calls_test;
  uint64_t calls_testall;
  uint64_t calls_waitany;
  uint64_t calls_waitsome;
} TraceStats;

static TraceStats stats = {0};
static double start_time = 0.0;
static int mpi_initialized = 0;

enum
{
  REQ_NONE = 0,
  REQ_P2P_SEND = 1,
  REQ_P2P_RECV = 2,
  REQ_COLLECTIVE = 3
};

typedef struct
{
  MPI_Request request;
  int active;
  int kind;
  uint64_t bytes;
} RequestInfo;

#define REQUEST_TABLE_SIZE 65536
static RequestInfo request_table[REQUEST_TABLE_SIZE];

static inline uint64_t datatype_nbytes(MPI_Datatype datatype, int count)
{
  int type_size = 0;
  PMPI_Type_size(datatype, &type_size);
  if (count < 0 || type_size < 0)
    return 0;
  return (uint64_t)type_size * (uint64_t)count;
}

static inline uint64_t datatype_nbytes_counts(MPI_Datatype datatype, const int *counts, int n)
{
  if (!counts || n <= 0)
    return 0;
  int type_size = 0;
  PMPI_Type_size(datatype, &type_size);
  if (type_size < 0)
    return 0;
  uint64_t total = 0;
  for (int i = 0; i < n; ++i)
    if (counts[i] > 0)
      total += (uint64_t)type_size * (uint64_t)counts[i];
  return total;
}

static inline unsigned int request_hash(MPI_Request request)
{
  const uintptr_t v = (uintptr_t)request;
  return (unsigned int)((v ^ (v >> 16U) ^ (v >> 32U)) & (REQUEST_TABLE_SIZE - 1));
}

static void request_store(MPI_Request request, int kind, uint64_t bytes)
{
  if (request == MPI_REQUEST_NULL)
    return;

  unsigned int idx = request_hash(request);
  for (unsigned int i = 0; i < REQUEST_TABLE_SIZE; ++i)
  {
    RequestInfo *slot = &request_table[(idx + i) & (REQUEST_TABLE_SIZE - 1)];
    if (!slot->active || slot->request == request)
    {
      slot->request = request;
      slot->kind = kind;
      slot->bytes = bytes;
      slot->active = 1;
      return;
    }
  }
}

static RequestInfo request_take(MPI_Request request)
{
  RequestInfo none = {MPI_REQUEST_NULL, 0, REQ_NONE, 0};
  if (request == MPI_REQUEST_NULL)
    return none;

  unsigned int idx = request_hash(request);
  for (unsigned int i = 0; i < REQUEST_TABLE_SIZE; ++i)
  {
    RequestInfo *slot = &request_table[(idx + i) & (REQUEST_TABLE_SIZE - 1)];
    if (!slot->active)
      return none;
    if (slot->request == request)
    {
      RequestInfo out = *slot;
      slot->active = 0;
      slot->request = MPI_REQUEST_NULL;
      slot->kind = REQ_NONE;
      slot->bytes = 0;
      return out;
    }
  }
  return none;
}

static void write_trace_file(void)
{
  int rank = -1;
  int size = -1;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  const char *prefix = getenv("MPI_TRACE_OUTPUT");
  char path[1024];
  if (prefix && prefix[0] != '\0')
    snprintf(path, sizeof(path), "%s_rank%d.csv", prefix, rank);
  else
    snprintf(path, sizeof(path), "/tmp/mpi_trace_rank%d.csv", rank);

  FILE *fp = fopen(path, "w");
  if (!fp)
    return;

  fprintf(fp, "rank,%d\n", rank);
  fprintf(fp, "size,%d\n", size);
  fprintf(fp, "pid,%d\n", getpid());
  fprintf(fp, "runtime_s,%.9f\n", stats.total_runtime);
  fprintf(fp, "collective_time_s,%.9f\n", stats.collective_time);
  fprintf(fp, "p2p_time_s,%.9f\n", stats.p2p_time);
  fprintf(fp, "wait_time_s,%.9f\n", stats.wait_time);
  fprintf(fp, "p2p_wait_time_s,%.9f\n", stats.p2p_wait_time);
  fprintf(fp, "collective_calls,%llu\n", (unsigned long long)stats.collective_calls);
  fprintf(fp, "p2p_calls,%llu\n", (unsigned long long)stats.p2p_calls);
  fprintf(fp, "wait_calls,%llu\n", (unsigned long long)stats.wait_calls);
  fprintf(fp, "collective_bytes,%llu\n", (unsigned long long)stats.collective_bytes);
  fprintf(fp, "p2p_send_bytes,%llu\n", (unsigned long long)stats.p2p_send_bytes);
  fprintf(fp, "p2p_recv_bytes,%llu\n", (unsigned long long)stats.p2p_recv_bytes);
  fprintf(fp, "calls_allgatherv,%llu\n", (unsigned long long)stats.calls_allgatherv);
  fprintf(fp, "calls_alltoallv,%llu\n", (unsigned long long)stats.calls_alltoallv);
  fprintf(fp, "calls_cancel,%llu\n", (unsigned long long)stats.calls_cancel);
  fprintf(fp, "calls_exscan,%llu\n", (unsigned long long)stats.calls_exscan);
  fprintf(fp, "calls_gatherv,%llu\n", (unsigned long long)stats.calls_gatherv);
  fprintf(fp, "calls_iallreduce,%llu\n", (unsigned long long)stats.calls_iallreduce);
  fprintf(fp, "calls_ibarrier,%llu\n", (unsigned long long)stats.calls_ibarrier);
  fprintf(fp, "calls_issend,%llu\n", (unsigned long long)stats.calls_issend);
  fprintf(fp, "calls_reduce_scatter,%llu\n", (unsigned long long)stats.calls_reduce_scatter);
  fprintf(fp, "calls_reduce_scatter_block,%llu\n", (unsigned long long)stats.calls_reduce_scatter_block);
  fprintf(fp, "calls_request_free,%llu\n", (unsigned long long)stats.calls_request_free);
  fprintf(fp, "calls_rsend,%llu\n", (unsigned long long)stats.calls_rsend);
  fprintf(fp, "calls_scan,%llu\n", (unsigned long long)stats.calls_scan);
  fprintf(fp, "calls_scatterv,%llu\n", (unsigned long long)stats.calls_scatterv);
  fprintf(fp, "calls_ssend,%llu\n", (unsigned long long)stats.calls_ssend);
  fprintf(fp, "calls_test,%llu\n", (unsigned long long)stats.calls_test);
  fprintf(fp, "calls_testall,%llu\n", (unsigned long long)stats.calls_testall);
  fprintf(fp, "calls_waitany,%llu\n", (unsigned long long)stats.calls_waitany);
  fprintf(fp, "calls_waitsome,%llu\n", (unsigned long long)stats.calls_waitsome);

  fclose(fp);
}

int MPI_Init(int *argc, char ***argv)
{
  const int ret = PMPI_Init(argc, argv);
  mpi_initialized = 1;
  start_time = PMPI_Wtime();
  return ret;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
  const int ret = PMPI_Init_thread(argc, argv, required, provided);
  mpi_initialized = 1;
  start_time = PMPI_Wtime();
  return ret;
}

int MPI_Finalize(void)
{
  if (mpi_initialized)
  {
    stats.total_runtime = PMPI_Wtime() - start_time;
    write_trace_file();
  }
  return PMPI_Finalize();
}

#define WRAP_COLLECTIVE(name, sig, args, bytes_expr) \
  int name sig                                         \
  {                                                    \
    const double t0 = PMPI_Wtime();                    \
    const int ret = P##name args;                      \
    const double dt = PMPI_Wtime() - t0;               \
    stats.collective_time += dt;                       \
    stats.collective_calls++;                          \
    stats.collective_bytes += (bytes_expr);            \
    return ret;                                        \
  }

#define WRAP_P2P_BLOCKING(name, sig, args, send_bytes_expr, recv_bytes_expr) \
  int name sig                                                                   \
  {                                                                              \
    const double t0 = PMPI_Wtime();                                              \
    const int ret = P##name args;                                                \
    const double dt = PMPI_Wtime() - t0;                                         \
    stats.p2p_time += dt;                                                        \
    stats.p2p_calls++;                                                           \
    stats.p2p_send_bytes += (send_bytes_expr);                                   \
    stats.p2p_recv_bytes += (recv_bytes_expr);                                   \
    return ret;                                                                  \
  }

WRAP_COLLECTIVE(MPI_Barrier, (MPI_Comm comm), (comm), 0)
WRAP_COLLECTIVE(MPI_Bcast,
                (void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm),
                (buffer, count, datatype, root, comm),
                datatype_nbytes(datatype, count))
WRAP_COLLECTIVE(MPI_Allreduce,
                (const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm),
                (sendbuf, recvbuf, count, datatype, op, comm),
                datatype_nbytes(datatype, count))
WRAP_COLLECTIVE(MPI_Reduce,
                (const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm),
                (sendbuf, recvbuf, count, datatype, op, root, comm),
                datatype_nbytes(datatype, count))
WRAP_COLLECTIVE(MPI_Allgather,
                (const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm),
                (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm),
                datatype_nbytes(sendtype, sendcount))
WRAP_COLLECTIVE(MPI_Gather,
                (const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm),
                (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm),
                datatype_nbytes(sendtype, sendcount))
WRAP_COLLECTIVE(MPI_Scatter,
                (const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm),
                (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm),
                datatype_nbytes(recvtype, recvcount))
WRAP_COLLECTIVE(MPI_Alltoall,
                (const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm),
                (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm),
                datatype_nbytes(sendtype, sendcount))

int MPI_Allgatherv(const void *sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void *recvbuf,
                   const int recvcounts[],
                   const int displs[],
                   MPI_Datatype recvtype,
                   MPI_Comm comm)
{
  stats.calls_allgatherv++;
  (void)displs;
  int size = 0;
  PMPI_Comm_size(comm, &size);
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes(sendtype, sendcount) + datatype_nbytes_counts(recvtype, recvcounts, size);
  return ret;
}

int MPI_Gatherv(const void *sendbuf,
                int sendcount,
                MPI_Datatype sendtype,
                void *recvbuf,
                const int recvcounts[],
                const int displs[],
                MPI_Datatype recvtype,
                int root,
                MPI_Comm comm)
{
  stats.calls_gatherv++;
  (void)displs;
  (void)root;
  int size = 0;
  PMPI_Comm_size(comm, &size);
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes(sendtype, sendcount) + datatype_nbytes_counts(recvtype, recvcounts, size);
  return ret;
}

int MPI_Scatterv(const void *sendbuf,
                 const int sendcounts[],
                 const int displs[],
                 MPI_Datatype sendtype,
                 void *recvbuf,
                 int recvcount,
                 MPI_Datatype recvtype,
                 int root,
                 MPI_Comm comm)
{
  stats.calls_scatterv++;
  (void)sendbuf;
  (void)displs;
  (void)root;
  int size = 0;
  PMPI_Comm_size(comm, &size);
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes_counts(sendtype, sendcounts, size) + datatype_nbytes(recvtype, recvcount);
  return ret;
}

int MPI_Alltoallv(const void *sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  MPI_Datatype sendtype,
                  void *recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  MPI_Datatype recvtype,
                  MPI_Comm comm)
{
  stats.calls_alltoallv++;
  (void)sendbuf;
  (void)sdispls;
  (void)recvbuf;
  (void)rdispls;
  int size = 0;
  PMPI_Comm_size(comm, &size);
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes_counts(sendtype, sendcounts, size) + datatype_nbytes_counts(recvtype, recvcounts, size);
  return ret;
}

int MPI_Reduce_scatter(const void *sendbuf,
                       void *recvbuf,
                       const int recvcounts[],
                       MPI_Datatype datatype,
                       MPI_Op op,
                       MPI_Comm comm)
{
  stats.calls_reduce_scatter++;
  (void)sendbuf;
  (void)recvbuf;
  (void)op;
  int size = 0;
  PMPI_Comm_size(comm, &size);
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes_counts(datatype, recvcounts, size);
  return ret;
}

int MPI_Reduce_scatter_block(const void *sendbuf,
                             void *recvbuf,
                             int recvcount,
                             MPI_Datatype datatype,
                             MPI_Op op,
                             MPI_Comm comm)
{
  stats.calls_reduce_scatter_block++;
  (void)sendbuf;
  (void)recvbuf;
  (void)op;
  int size = 0;
  PMPI_Comm_size(comm, &size);
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes(datatype, recvcount * size);
  return ret;
}

int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  stats.calls_scan++;
  (void)sendbuf;
  (void)recvbuf;
  (void)op;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes(datatype, count);
  return ret;
}

int MPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  stats.calls_exscan++;
  (void)sendbuf;
  (void)recvbuf;
  (void)op;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes(datatype, count);
  return ret;
}

WRAP_P2P_BLOCKING(MPI_Send,
                  (const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm),
                  (buf, count, datatype, dest, tag, comm),
                  datatype_nbytes(datatype, count),
                  0)
WRAP_P2P_BLOCKING(MPI_Recv,
                  (void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status),
                  (buf, count, datatype, source, tag, comm, status),
                  0,
                  datatype_nbytes(datatype, count))
WRAP_P2P_BLOCKING(MPI_Sendrecv,
                  (const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                   MPI_Comm comm, MPI_Status *status),
                  (sendbuf, sendcount, sendtype, dest, sendtag,
                   recvbuf, recvcount, recvtype, source, recvtag, comm, status),
                  datatype_nbytes(sendtype, sendcount),
                  datatype_nbytes(recvtype, recvcount))

int MPI_Isend(const void *buf,
              int count,
              MPI_Datatype datatype,
              int dest,
              int tag,
              MPI_Comm comm,
              MPI_Request *request)
{
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
  const double dt = PMPI_Wtime() - t0;
  stats.p2p_time += dt;
  stats.p2p_calls++;
  const uint64_t bytes = datatype_nbytes(datatype, count);
  stats.p2p_send_bytes += bytes;
  if (ret == MPI_SUCCESS && request)
    request_store(*request, REQ_P2P_SEND, bytes);
  return ret;
}

int MPI_Irecv(void *buf,
              int count,
              MPI_Datatype datatype,
              int source,
              int tag,
              MPI_Comm comm,
              MPI_Request *request)
{
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
  const double dt = PMPI_Wtime() - t0;
  stats.p2p_time += dt;
  stats.p2p_calls++;
  const uint64_t bytes = datatype_nbytes(datatype, count);
  stats.p2p_recv_bytes += bytes;
  if (ret == MPI_SUCCESS && request)
    request_store(*request, REQ_P2P_RECV, bytes);
  return ret;
}

int MPI_Issend(const void *buf,
               int count,
               MPI_Datatype datatype,
               int dest,
               int tag,
               MPI_Comm comm,
               MPI_Request *request)
{
  stats.calls_issend++;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
  const double dt = PMPI_Wtime() - t0;
  stats.p2p_time += dt;
  stats.p2p_calls++;
  const uint64_t bytes = datatype_nbytes(datatype, count);
  stats.p2p_send_bytes += bytes;
  if (ret == MPI_SUCCESS && request)
    request_store(*request, REQ_P2P_SEND, bytes);
  return ret;
}

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  stats.calls_ssend++;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Ssend(buf, count, datatype, dest, tag, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.p2p_time += dt;
  stats.p2p_calls++;
  stats.p2p_send_bytes += datatype_nbytes(datatype, count);
  return ret;
}

int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  stats.calls_rsend++;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Rsend(buf, count, datatype, dest, tag, comm);
  const double dt = PMPI_Wtime() - t0;
  stats.p2p_time += dt;
  stats.p2p_calls++;
  stats.p2p_send_bytes += datatype_nbytes(datatype, count);
  return ret;
}

int MPI_Iallreduce(const void *sendbuf,
                   void *recvbuf,
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPI_Comm comm,
                   MPI_Request *request)
{
  stats.calls_iallreduce++;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  stats.collective_bytes += datatype_nbytes(datatype, count);
  if (ret == MPI_SUCCESS && request)
    request_store(*request, REQ_COLLECTIVE, datatype_nbytes(datatype, count));
  return ret;
}

int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request)
{
  stats.calls_ibarrier++;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Ibarrier(comm, request);
  const double dt = PMPI_Wtime() - t0;
  stats.collective_time += dt;
  stats.collective_calls++;
  if (ret == MPI_SUCCESS && request)
    request_store(*request, REQ_COLLECTIVE, 0);
  return ret;
}

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{
  MPI_Request tracked = MPI_REQUEST_NULL;
  if (request)
    tracked = *request;

  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Wait(request, status);
  const double dt = PMPI_Wtime() - t0;
  stats.wait_time += dt;
  stats.wait_calls++;

  const RequestInfo info = request_take(tracked);
  if (info.active && (info.kind == REQ_P2P_SEND || info.kind == REQ_P2P_RECV))
    stats.p2p_wait_time += dt;

  return ret;
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
{
  MPI_Request *before = NULL;
  if (count > 0)
  {
    before = (MPI_Request *)malloc(sizeof(MPI_Request) * (size_t)count);
    if (before)
      memcpy(before, array_of_requests, sizeof(MPI_Request) * (size_t)count);
  }

  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Waitall(count, array_of_requests, array_of_statuses);
  const double dt = PMPI_Wtime() - t0;
  stats.wait_time += dt;
  stats.wait_calls++;

  if (before)
  {
    int p2p_count = 0;
    for (int i = 0; i < count; ++i)
    {
      const RequestInfo info = request_take(before[i]);
      if (info.active && (info.kind == REQ_P2P_SEND || info.kind == REQ_P2P_RECV))
        p2p_count++;
    }
    if (p2p_count > 0)
      stats.p2p_wait_time += dt;
    free(before);
  }
  return ret;
}

int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status)
{
  stats.calls_waitany++;
  MPI_Request *before = NULL;
  if (count > 0)
  {
    before = (MPI_Request *)malloc(sizeof(MPI_Request) * (size_t)count);
    if (before)
      memcpy(before, array_of_requests, sizeof(MPI_Request) * (size_t)count);
  }
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Waitany(count, array_of_requests, index, status);
  const double dt = PMPI_Wtime() - t0;
  stats.wait_time += dt;
  stats.wait_calls++;
  if (before && index && *index != MPI_UNDEFINED && *index >= 0 && *index < count)
  {
    const RequestInfo info = request_take(before[*index]);
    if (info.active && (info.kind == REQ_P2P_SEND || info.kind == REQ_P2P_RECV))
      stats.p2p_wait_time += dt;
  }
  free(before);
  return ret;
}

int MPI_Waitsome(int incount,
                 MPI_Request array_of_requests[],
                 int *outcount,
                 int array_of_indices[],
                 MPI_Status array_of_statuses[])
{
  stats.calls_waitsome++;
  MPI_Request *before = NULL;
  if (incount > 0)
  {
    before = (MPI_Request *)malloc(sizeof(MPI_Request) * (size_t)incount);
    if (before)
      memcpy(before, array_of_requests, sizeof(MPI_Request) * (size_t)incount);
  }
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
  const double dt = PMPI_Wtime() - t0;
  stats.wait_time += dt;
  stats.wait_calls++;
  if (before && outcount && array_of_indices && *outcount > 0)
  {
    int p2p_count = 0;
    for (int i = 0; i < *outcount; ++i)
    {
      const int idx = array_of_indices[i];
      if (idx >= 0 && idx < incount)
      {
        const RequestInfo info = request_take(before[idx]);
        if (info.active && (info.kind == REQ_P2P_SEND || info.kind == REQ_P2P_RECV))
          p2p_count++;
      }
    }
    if (p2p_count > 0)
      stats.p2p_wait_time += dt;
  }
  free(before);
  return ret;
}

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
  stats.calls_test++;
  MPI_Request tracked = MPI_REQUEST_NULL;
  if (request)
    tracked = *request;
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Test(request, flag, status);
  const double dt = PMPI_Wtime() - t0;
  stats.wait_time += dt;
  stats.wait_calls++;
  if (ret == MPI_SUCCESS && flag && *flag)
  {
    const RequestInfo info = request_take(tracked);
    if (info.active && (info.kind == REQ_P2P_SEND || info.kind == REQ_P2P_RECV))
      stats.p2p_wait_time += dt;
  }
  return ret;
}

int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[])
{
  stats.calls_testall++;
  MPI_Request *before = NULL;
  if (count > 0)
  {
    before = (MPI_Request *)malloc(sizeof(MPI_Request) * (size_t)count);
    if (before)
      memcpy(before, array_of_requests, sizeof(MPI_Request) * (size_t)count);
  }
  const double t0 = PMPI_Wtime();
  const int ret = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
  const double dt = PMPI_Wtime() - t0;
  stats.wait_time += dt;
  stats.wait_calls++;
  if (ret == MPI_SUCCESS && flag && *flag && before)
  {
    int p2p_count = 0;
    for (int i = 0; i < count; ++i)
    {
      const RequestInfo info = request_take(before[i]);
      if (info.active && (info.kind == REQ_P2P_SEND || info.kind == REQ_P2P_RECV))
        p2p_count++;
    }
    if (p2p_count > 0)
      stats.p2p_wait_time += dt;
  }
  free(before);
  return ret;
}

int MPI_Request_free(MPI_Request *request)
{
  stats.calls_request_free++;
  if (request)
    request_take(*request);
  return PMPI_Request_free(request);
}

int MPI_Cancel(MPI_Request *request)
{
  stats.calls_cancel++;
  if (request)
    request_take(*request);
  return PMPI_Cancel(request);
}
