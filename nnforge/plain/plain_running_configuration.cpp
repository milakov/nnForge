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

#include "plain_running_configuration.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nnforge
{
	namespace plain
	{
		plain_running_configuration::plain_running_configuration(
			int openmp_thread_count,
			float max_memory_usage_gigabytes)
			: openmp_thread_count(openmp_thread_count)
			, max_memory_usage_gigabytes(max_memory_usage_gigabytes)
		{
			#ifndef _OPENMP
			this->openmp_thread_count = 1;
			#endif
		}

		unsigned int plain_running_configuration::get_max_entry_count(
			const buffer_plain_size_configuration& buffers_config,
			float ratio) const
		{
			size_t memory_left = static_cast<size_t>(max_memory_usage_gigabytes * ratio * static_cast<float>(1 << 30)) - buffers_config.constant_buffer_size;
			size_t entry_count_limited_by_global = memory_left / buffers_config.per_entry_buffer_size;

			return static_cast<unsigned int>(entry_count_limited_by_global);
		}

		std::ostream& operator<< (std::ostream& out, const plain_running_configuration& running_configuration)
		{
			out << "--- Configuration ---" << std::endl;
			#ifdef _OPENMP
			out << "Max OpenMP thread count = " << omp_get_max_threads() << std::endl;
			#else
			out << "Compiled without OpenMP support"
			#endif

			out << "--- Settings ---" << std::endl;

			out << "Max memory usage = " << running_configuration.max_memory_usage_gigabytes << " GB" << std::endl;
			out << "OpenMP thread count = " << running_configuration.openmp_thread_count << std::endl;

			return out;
		}
	}
}
