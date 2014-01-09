/*
 *  Copyright 2011-2014 Maxim Milakov
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
		struct supervised_data_reader_functor
		{
			supervised_data_reader_functor();

			supervised_data_reader_functor(
				unsigned int entries_to_read_count,
				supervised_data_reader * reader,
				unsigned char * input,
				float * output,
				void * d_input,
				void * d_output,
				cudaStream_t stream);

			supervised_data_reader_functor& operator =(const supervised_data_reader_functor& other);

			unsigned int operator()();

			unsigned int entries_to_read_count;
			supervised_data_reader * reader;
			unsigned char * input;
			float * output;
			void * d_input;
			void * d_output;
			cudaStream_t stream;

			std::string * error;
		};

		class supervised_data_reader_async_helper
		{
		public:
			supervised_data_reader_async_helper();

			supervised_data_reader_async_helper(const supervised_data_reader_functor& fun);

			virtual ~supervised_data_reader_async_helper();

			void start();

			unsigned int wait();

			unsigned int operator()();

			supervised_data_reader_functor fun;

			void * impl;

		private:
			std::string error;
		};
	}
}
