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

#pragma once

#include "../supervised_data_reader.h"

#include <string>
#include <cuda_runtime.h>

namespace nnforge
{
	namespace cuda
	{
		struct training_data_reader_functor
		{
			training_data_reader_functor();

			training_data_reader_functor(
				unsigned int entries_to_read_count,
				supervised_data_reader * reader,
				unsigned char * input,
				float * output,
				void * d_input,
				void * d_output,
				unsigned int input_neuron_count,
				unsigned int output_neuron_count,
				size_t input_neuron_elem_size,
				cudaStream_t stream);

			training_data_reader_functor& operator =(const training_data_reader_functor& other);

			unsigned int operator()();

			unsigned int entries_to_read_count;
			supervised_data_reader * reader;
			unsigned char * input;
			float * output;
			void * d_input;
			void * d_output;
			unsigned int input_neuron_count;
			unsigned int output_neuron_count;
			size_t input_neuron_elem_size;
			cudaStream_t stream;

			std::string * error;
		};

		class training_data_reader_helper
		{
		public:
			training_data_reader_helper();

			training_data_reader_helper(const training_data_reader_functor& fun);

			virtual ~training_data_reader_helper();

			void start();

			unsigned int wait();

			unsigned int operator()();

			training_data_reader_functor fun;

			void * impl;

		private:
			std::string error;
		};
	}
}
