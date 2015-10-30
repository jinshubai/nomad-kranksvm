#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <vector>
#include "scheduler.h"

 Scheduler::Scheduler(const struct rksvm_problem *prob, const struct rksvm_parameter *param)
 {
	 int i, rankid;
	 this->l = prob->l;
	 this->nr_ranks = param->nr_ranks;
	 this->prob = prob;
	 this->start_ptr = new int[nr_ranks];
	 this->local_l = new int[nr_ranks];
	 this->length_of_each_rksvm_node = prob->length_of_each_rksvm_node;
	 this->nr_send = new int[nr_ranks];
	 this->nr_recv = new int[nr_ranks];

	 //compute *start_ptr
	 MPI_Allgather(&l, 1, MPI_INT, local_l, 1, MPI_INT, MPI_COMM_WORLD);
	 start_ptr[0] = 0;
	 for(i=1;i<nr_ranks;i++)
		 start_ptr[i] = start_ptr[i-1] + local_l[i-1];
	 
	 //generate a copy for each rksvm_node in the current machine
	 for(i=0;i<l;i++)
	 {
		 data_node *copy_x = callocator<data_node>().allocate(1);
		 copy_x->first_time = true;
		 copy_x->length = length_of_each_rksvm_node[i];
		 MPI_Comm_rank(MPI_COMM_WORLD, &copy_x->initial_rank);
		 MPI_Comm_rank(MPI_COMM_WORLD, &copy_x->current_rank);
		 copy_x->global_index = i + start_ptr[copy_x->initial_rank];
		 copy_x->x = callocator<rksvm_node>().allocate(copy_x->length);
		 memcpy(copy_x->x, prob->x[i],(size_t)copy_x->length*sizeof(rksvm_node));
		 push(copy_x);
	 }
	 //computing *nr_send and *nr_recv
	 //we hope that a cycle sending or receiving can be achieved.
	 //so, it can detemine how many copies should be sended or received
	 if(nr_ranks>1)
	 {
		 for(rankid=0;rankid<nr_ranks;rankid++)
		 {
			 nr_send[rankid] = 0;
			 for(i=0;i<nr_ranks;i++)
			 {
				 if((rankid+1)%nr_ranks!=i)
					 nr_send[rankid] += local_l[i];
			 }
			 nr_recv[(rankid+1)%nr_ranks] = nr_send[rankid];
		 }
	 }
 }

Scheduler::~Scheduler()
{
	delete[] start_ptr;
	delete[] local_l;
	delete[] nr_send;
	delete[] nr_recv;
	while(true)
	{
		data_node *copy_x = nullptr;
		bool success = queue_.try_pop(copy_x);
		if(success)
		{
			int lth = copy_x->length;
			callocator<rksvm_node>().deallocate(copy_x->x,lth);
			callocator<data_node>().destroy(copy_x);
			callocator<data_node>().deallocate(copy_x,1);
		 }
		 else
		 {
			 break;
		 }
	 }
}

void Scheduler::push(data_node *copy_x)
 {
	 queue_.push(copy_x);
 }


data_node* Scheduler::pop()
 {
	 data_node *copy_x = nullptr;
	 bool success = queue_.try_pop(copy_x);
	 if(success)
		 return copy_x;
	 else
		 return nullptr;
 }