# VBASE environment installation and script testing

## Environment installation

### Download VBASE source code

```
git clone https://github.com/microsoft/MSVBASE.git
cd MSVBASE
git submodule update --init --recursive
./scripts/patch.sh
```

### Customize the ef_search parameter

Before starting to build the image, you need to modify the source code so that we can customize the search parameter ef_search.

Step 1: Locate src/lib.cpp

```c++
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

Step 2: In src/hnswindex.cpp

```c++
// Declare external variable
extern int hnsw_ef_search_value;

// In the hnsw_gettuple function, replace the content in if (scanState->first) with the following code
if (scanState->first)
{
   scanState->workSpace = new HNSWIndexScan::WorkSpace();
   std::string path = std::string(DataDir) + std::string("/") +
                    std::string(DatabasePath) + std::string("/") +
                    std::string(RelationGetRelationName(scan->indexRelation));
    int ef_search = hnsw_ef_search_value;
    if (scan->orderByData == NULL)
    {
        if (scan->keyData == NULL)
            return false;
        if (scan->keyData->sk_flags & SK_ISNULL)
            return false;

        Datum value = scan->keyData->sk_argument;
	    scanState->workSpace->array = convert_array_to_vector(value);
        scanState->hasRangeFilter = true;
        scanState->inRange = false;
        scanState->range = scanState->workSpace->array[0];
        scanState->workSpace->resultIterator =
                   HNSWIndexScan::BeginScan((char *)(scanState->workSpace->array.data() + 1),path);
    }
    else{
        if (scan->orderByData->sk_flags & SK_ISNULL)
            return false;
        Datum value = scan->orderByData->sk_argument;
	    scanState->workSpace->array = convert_array_to_vector(value);
        scanState->hasRangeFilter = false;
	 	scanState->range = ef_search;
        scanState->workSpace->resultIterator =
                   HNSWIndexScan::BeginScan((char *)scanState->workSpace->array.data(),path);
    }
    scan->xs_inorder = false;
    scanState->first = false;
}

```

### Build the image and run it

```bash
./scripts/dockerbuild.sh
```

```bash
./scripts/dockerrun.sh
```

## Script execution

### Modify the configuration file

Enter the Docker container

```bash
sudo docker exec -it vbase_open_source bash
```

Modify /u02/pgdata/13/pg_hba.conf to enable remote connections

```bash
vim /u02/pgdata/13/pg_hba.conf
```

Add at the last line

```
host    all             all             0.0.0.0/0               trust
```

restart

```bash
pg_ctl reload -D /u02/pgdata/13
```

### Install Python Environment

The Python version is 3.8.10

Download some necessary Python libraries

```
pip install psycopg2
```

Each dataset corresponds to a directory, and the test scripts are divided into a single-threaded test script (including table creation and index creation) and a multi-threaded test script.

### run script

Before testing, you need to install the vectordb plugin.

```sql
docker exec -it vbase_open_source bash 
psql -U vectordb
create extension vectordb;
```

Change the host in the script to the IP of your own VBASE container. 
The paths for the dataset and labels also need to be changed to the paths where you have installed them.