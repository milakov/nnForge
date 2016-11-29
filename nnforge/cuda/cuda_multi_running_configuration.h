/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include <vector>
#include <memory>
#include <ostream>

#include "cuda_running_configuration.h"
#include "../threadpool_job_runner.h"

namespace nnforge
{
	namespace cuda
	{
		class cuda_multi_running_configuration
		{
		public:
			typedef std::shared_ptr<cuda_multi_running_configuration> ptr;
			typedef std::shared_ptr<const cuda_multi_running_configuration> const_ptr;

			cuda_multi_running_configuration(
				const std::vector<unsigned int>& device_id_list,
				float max_global_memory_usage_ratio,
				unsigned int reserved_thread_count,
				bool dont_share_buffers,
				bool single_command_stream,
				unsigned int optimize_action_graph_assumed_chunk_size,
				float cuda_fixed_working_buffers_ratio);

			~cuda_multi_running_configuration() = default;

			size_t get_max_fixed_working_buffers_size() const;

			bool is_dont_share_buffers() const;

			bool is_single_command_stream() const;

			threadpool_job_runner::ptr get_job_runner() const;

			static std::vector<unsigned int> get_default_device_id_list();

		public:
			std::vector<cuda_running_configuration::ptr> cuda_config_list;

			int driver_version;
			int runtime_version;
			int cuda_device_count;

			unsigned int reserved_thread_count;
			bool dont_share_buffers;
			bool single_command_stream;
			unsigned int optimize_action_graph_assumed_chunk_size;

		private:
			threadpool_job_runner::ptr job_runner;

		private:
			void update_parameters();

			cuda_multi_running_configuration() = delete;
			cuda_multi_running_configuration(const cuda_multi_running_configuration&) = delete;
			cuda_multi_running_configuration& operator =(const cuda_multi_running_configuration&) = delete;
		};

		std::ostream& operator<< (std::ostream& out, const cuda_multi_running_configuration& multi_running_configuration);
	}
}
