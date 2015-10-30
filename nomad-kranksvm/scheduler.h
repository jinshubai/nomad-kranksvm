#ifndef SCHEDULER_H_
#define SCHEDULER_H_
#include <vector>
#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <tbb/atomic.h>
#include <tbb/tbb.h>
#include <tbb/compat/thread>
#include "rksvm.h"

template <typename T>
using sallocator = tbb::scalable_allocator<T>;
template <typename T>
using callocator = std::allocator<T>;
template <typename T>
using atomic = tbb::atomic<T>;
using tbb::tick_count;

class Scheduler
{
   public:
	   Scheduler(const struct rksvm_problem *prob, const struct rksvm_parameter *param);
	   ~Scheduler();
	   void push(data_node *copy_x);
	   data_node *pop();
	   int *nr_send;
	   int *nr_recv;
	   int *local_l;
	   int *start_ptr;
   private:
	  int l;
	  int nr_ranks;
	  const struct rksvm_problem *prob;
	  int *length_of_each_rksvm_node;
	  tbb::concurrent_queue<data_node*, callocator<data_node*>> queue_;
};
#endif