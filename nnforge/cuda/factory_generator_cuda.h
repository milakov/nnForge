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

#include "../factory_generator.h"
#include "cuda_multi_running_configuration.h"

namespace nnforge
{
	namespace cuda
	{
		class factory_generator_cuda : public factory_generator
		{
		public:
			factory_generator_cuda(
				const std::string& cuda_device_id_list_str,
				float cuda_max_global_memory_usage_ratio,
				unsigned int cuda_reserved_thread_count,
				bool cuda_dont_share_buffers,
				bool cuda_single_command_stream,
				unsigned int cuda_optimize_action_graph_assumed_chunk_size,
				float cuda_fixed_working_buffers_ratio,
				const std::string& communicator_type);

			factory_generator_cuda() = default;

			~factory_generator_cuda() = default;

			virtual void initialize();

			virtual forward_propagation_factory::ptr create_forward_propagation_factory() const;

			virtual backward_propagation_factory::ptr create_backward_propagation_factory() const;

			virtual void info() const;

			virtual std::vector<float_option> get_float_options();

			virtual std::vector<int_option> get_int_options();

			virtual std::vector<bool_option> get_bool_options();

			virtual std::vector<string_option> get_string_options();

		protected:
			float cuda_max_global_memory_usage_ratio;
			int cuda_reserved_thread_count;
			bool cuda_dont_share_buffers;
			bool cuda_single_command_stream;
			int cuda_optimize_action_graph_assumed_chunk_size;
			float cuda_fixed_working_buffers_ratio;
			std::string cuda_device_id_list_str;
			std::string communicator_type;

			cuda_multi_running_configuration::const_ptr cuda_multi_config;
		};
	}
}
