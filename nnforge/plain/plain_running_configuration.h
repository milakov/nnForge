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

#include <ostream>

#include "buffer_plain_size_configuration.h"

#include "../nn_types.h"

namespace nnforge
{
	namespace plain
	{
		class plain_running_configuration
		{
		public:
			typedef nnforge_shared_ptr<const plain_running_configuration> const_ptr;

			plain_running_configuration(
				int openmp_thread_count,
				float max_memory_usage_gigabytes);

			unsigned int get_max_entry_count(
				const buffer_plain_size_configuration& buffers_config,
				float ratio = 1.0F) const;

			float max_memory_usage_gigabytes;
			int openmp_thread_count;

		private:
			plain_running_configuration();
			plain_running_configuration(const plain_running_configuration&);
			plain_running_configuration& operator =(const plain_running_configuration&);
		};

		std::ostream& operator<< (std::ostream& out, const plain_running_configuration& running_configuration);
	}
}
