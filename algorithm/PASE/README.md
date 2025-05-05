# PASE environment installation and script testing

## Environment installation

### install pg11

Install PostgreSQL 11 (the official PASE version only supports PostgreSQL 11)

### Download PASE source code

```bash
git clone https://github.com/alipay/PASE.git
```

### Customize the ef_search parameter

Before starting to make, you need to modify the source code so that we can customize the search parameter ef_search.

Step 1: Locate pase_handler.c

```c
// Add header file
#include "utils/guc.h"

// Add global variable
int hnsw_ef_search_value = 86;

// Add in the _PG_init function
DefineCustomIntVariable(
    "hnsw.ef_search",
    "Sets the ef_search parameter for HNSW.", 
    "Valid values are positive integers.",    
    &hnsw_ef_search_value,           
    86,                              
    1,                               
    1000,                            
    PGC_USERSET,                     
    0,                               
    NULL,                            
    NULL,                            
    NULL                             
);
```

Step 2: hnsw/hnsw_scan.c

```c
// Declare external variable
extern int hnsw_ef_search_value;

// In the HNSWSearch function， replace with the following code
static void
HNSWSearch(Relation index, HNSWScanOpaque so, HNSWMetaPage meta, uint16 topk,
  float4 *queryVec) {
  HNSWGlobalId nearest;
  float dNearest;
  int ef;
  const int level = 1;
  bool type;
  PriorityQueue *result;
  HNSWOptions *opts = &meta->opts;
  HNSWVtable vtable;
  HNSWPriorityQueueNode* node;

  if (HNSW_CHECK_GID(meta->entry_gid)) {
    return;
  }
  nearest = meta->entry_gid;
  dNearest = Distance(index, opts, queryVec, nearest);
  GreedyUpdateNearest(index, opts, opts->real_max_level, level, &nearest, &dNearest, queryVec);
  int ef_search = hnsw_ef_search_value;
  ef = ef_search;
  elog(WARNING, "ef: %d", ef);
  HVTInit(index->rd_indexcxt, &vtable);
  type = true; // farthest --> nearest
  result = PriorityQueueAllocate(HNSWPriorityQueueCmp, &type);

  // do search on level 0
  HNSWDoSearch(index, opts, ef, queryVec, nearest, dNearest, 0, result, &vtable);
  // remove resident node
  while (PriorityQueueSize(result) > topk) {
    node = (HNSWPriorityQueueNode *)PriorityQueuePop(result);
    pfree(node);
  }

  // MaxHeap to MinHeap
  while (PriorityQueueSize(result) > 0) {
    node = (HNSWPriorityQueueNode *)PriorityQueuePop(result); 
    PriorityQueueAdd(so->queue, (PriorityQueueNode *)node);
  }
  PriorityQueueFree(result);
  HVTFree(&vtable);
}
```

### make

```bash
make USE_PGXS=1
```

After running `make USE_PGXS=1`, under normal circumstances, it will place three files in their respective locations. If they are not placed in the correct locations, you need to manually copy them.

```bash
sudo cp pase.so /usr/lib/postgresql/11/lib/  
sudo cp pase.control /usr/share/postgresql/11/extension/  
sudo cp pase--0.0.1.sql /usr/share/postgresql/11/extension/  
```

### Enter PostgreSQL to create the extension

```sql
sudo -u postgres -i
psql
create extension pase;
-- Check if the installation was successful
\dx
```



## Script execution

### Modify the configuration file

```bash
sudo vim /etc/postgresql/11/main/pg_hba.conf
```

Add at the last line

```
host    all             all             0.0.0.0/0               trust
```

```bash
sudo vim /etc/postgresql/11/main/postgresql.conf
listen_addresses = '*'
```

restart

```bash
sudo systemctl restart postgresql
```

### Install Python Environment

The Python version is 3.8.10

Download some necessary Python libraries

```
pip install psycopg2
```

Each dataset corresponds to a directory, and the test scripts are divided into a single-threaded test script (including table creation and index creation) and a multi-threaded test script.

Change the host in the script to the IP of your own PASE.

The paths for the dataset and labels also need to be changed to the paths where you have installed them.