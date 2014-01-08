/*
 *  Copyright 2011-2013 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "training_data_reader_helper.h"

#include "util_cuda.h"
#include "cuda_profiling.h"
#include "neural_network_cuda_exception.h"

#include <boost/thread.hpp>

namespace nnforge
{
	namespace cuda
	{
		struct thread_struct
		{
			boost::packaged_task<unsigned int> enqueue_copy_training_data_task;
			boost::unique_future<unsigned int> fu;
			boost::thread t;
		};

		training_data_reader_functor::training_data_reader_functor()
		{
		}

		training_data_reader_functor::training_data_reader_functor(
			unsigned int entries_to_read_count,
			supervised_data_reader * reader,
			unsigned char * input,
			float * output,
			void * d_input,
			void * d_output,
			unsigned int input_neuron_count,
			unsigned int output_neuron_count,
			size_t input_neuron_elem_size,
			cudaStream_t stream)
			: entries_to_read_count(entries_to_read_count)
			, reader(reader)
			, input(input)
			, output(output)
			, d_input(d_input)
			, d_output(d_output)
			, input_neuron_count(input_neuron_count)
			, output_neuron_count(output_neuron_count)
			, input_neuron_elem_size(input_neuron_elem_size)
			, stream(stream)
		{
		}

		training_data_reader_functor& training_data_reader_functor::operator =(const training_data_reader_functor& other)
		{
			this->entries_to_read_count = other.entries_to_read_count;
			this->reader = other.reader;
			this->input = other.input;
			this->output = other.output;
			this->d_input = other.d_input;
			this->d_output = other.d_output;
			this->input_neuron_count = other.input_neuron_count;
			this->output_neuron_count = other.output_neuron_count;
			this->input_neuron_elem_size = other.input_neuron_elem_size;
			this->stream = other.stream;

			return *this;
		}

		unsigned int training_data_reader_functor::operator()()
		{
			unsigned int entries_read_count = 0;
			try
			{
				PUSH_RANGE("Reading training data", 0)
				while(entries_read_count < entries_to_read_count)
				{
					bool entry_read = reader->read(
						input + (input_neuron_count * entries_read_count * input_neuron_elem_size),
						output + (output_neuron_count * entries_read_count));

					if (!entry_read)
						break;

					entries_read_count++;
				}
				POP_RANGE

				cuda_safe_call(cudaMemcpyAsync(
					d_input,
					input,
					entries_read_count * input_neuron_count * input_neuron_elem_size,
					cudaMemcpyHostToDevice,
					stream));
				cuda_safe_call(cudaMemcpyAsync(
					d_output,
					output,
					entries_read_count * output_neuron_count * sizeof(float),
					cudaMemcpyHostToDevice,
					stream));
			}
			catch (std::runtime_error& e)
			{
				*error = e.what();
			}

			return entries_read_count;
		}

		training_data_reader_helper::training_data_reader_helper()
			: impl(0)
		{
		}

		training_data_reader_helper::training_data_reader_helper(const training_data_reader_functor& fun)
			: impl(0)
			, fun(fun)
		{
		}

		training_data_reader_helper::~training_data_reader_helper()
		{
			if (impl != 0)
				delete ((thread_struct *)impl);
		}

		void training_data_reader_helper::start()
		{
			error.clear();
			fun.error = &error;

			if (impl != 0)
				delete ((thread_struct *)impl);

			thread_struct * tst = new thread_struct();
			impl = tst;
			tst->enqueue_copy_training_data_task = boost::packaged_task<unsigned int>(fun);
			tst->fu = tst->enqueue_copy_training_data_task.get_future();
			tst->t = boost::thread(boost::move(tst->enqueue_copy_training_data_task));
		}

		unsigned int training_data_reader_helper::wait()
		{
			thread_struct * tst = static_cast<thread_struct *>(impl);
			tst->fu.wait();
			unsigned int res = tst->fu.get();

			if (!error.empty())
				throw std::runtime_error(error);

			return res;
		}
	}
}
