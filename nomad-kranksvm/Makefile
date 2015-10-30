CXX = mpic++
CC = mpicc
CFLAGS = -fPIC -O3 -fopenmp -std=c++11 -Wall -fpermissive -g -gstabs+
CBLAS = blas/blas.a
LOGFLAGS=-DLOG

# TBB related paths
TBBROOT = tbb
TBBLIB = -L${TBBROOT}/lib -ltbb -ltbbmalloc

INCLUDEPATH = -I${TBBROOT}/include
LIBS = ${TBBLIB} -lrt -lm

all: train predict split

split: split.o
	${CXX} ${CFLAGS} -o split split.o 

predict: predict.c rksvm.o scheduler.o tron.o selectiontree.o $(CBLAS)
	$(CXX) $(CFLAGS) $+ ${INCLUDEPATH} -o predict -lm ${LIBS}

train: train.c rksvm.o scheduler.o tron.o selectiontree.o $(CBLAS)
	$(CXX) $(CFLAGS) $+ ${INCLUDEPATH} -o train -lm ${LIBS}

rksvm.o: rksvm.cpp rksvm.h
	$(CXX) $(CFLAGS) ${INCLUDEPATH} -c rksvm.cpp ${LIBS}

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

selectiontree.o: selectiontree.cpp selectiontree.h
	$(CXX) $(CFLAGS) -c -o selectiontree.o selectiontree.cpp

scheduler.o: scheduler.cpp scheduler.h
	$(CXX) $(CFLAGS) ${INCLUDEPATH} -c -o scheduler.o scheduler.cpp ${LIBS}

split.o: split.cpp split.h
	${CXX} ${CFLAGS} -c -o split.o split.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';
clean:
	make -C blas clean
	rm -f *.o train predict split