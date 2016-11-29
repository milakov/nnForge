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

#include "../forward_propagation.h"
#include "cuda_multi_running_configuration.h"
#include "layer_testing_schema.h"
#include "cuda_stream.h"
#include "cuda_event.h"

#include <map>
#include <condition_variable>
#include <mutex>

namespace nnforge
{
	namespace cuda
	{
		class forward_propagation_cuda : public forward_propagation
		{
		public:
			forward_propagation_cuda(
				const network_schema& schema,
				const std::vector<std::string>& output_layer_names,
				debug_state::ptr debug,
				profile_state::ptr profile,
				cuda_multi_running_configuration::const_ptr cuda_multi_config);

			virtual ~forward_propagation_cuda() = default;

		protected:
			// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
			virtual void actual_set_data(network_data::const_ptr data);

			virtual void actual_clear_data();

			// schema, network data and data are guaranteed to be compatible
			virtual void actual_run(
				structured_data_bunch_reader& reader,
				structured_data_bunch_writer& writer,
				unsigned int& entries_processed,
				std::map<layer_name_with_action, float>& action_seconds);

			// The method is called when client calls set_input_configuration_specific and the configuration is modified.
			// The layer_config_map is guaranteed to be compatible with schema
			virtual void layer_config_map_modified();

			virtual float get_max_flops() const;

		private:
			void setup_network_cuda();

			void update_data();

			void setup_layer_buffer_sizes();

			void setup_dedicated_buffer_sizes();

			void setup_io_host_buffer_sizes();

			void setup_temporary_working_fixed_buffer_sizes();

			void setup_streams_and_events();

			void update_max_entry_count();

			void setup_optimized_action_schema();

		private:
			class run_kernels_params
			{
			public:
				run_kernels_params(
					int device_pos,
					std::map<std::string, std::array<cuda_linear_buffer_device::ptr, 2> >& dedicated_buffers,
					unsigned int current_max_entry_count);

				std::map<std::string, std::array<cuda_linear_buffer_device::ptr, 2> >& dedicated_buffers;
				unsigned int current_max_entry_count;

				std::map<layer_name_with_action, double> action_seconds;

				std::string error_message;

				unsigned int run_kernels_thread_entry_to_process_count;

				bool run_kernels_finished;
				std::mutex run_kernels_finished_mutex;
				std::condition_variable run_kernels_finished_condition;

				bool run_kernels_task_ready;
				std::mutex run_kernels_pending_mutex;
				std::condition_variable run_kernels_pending_condition;

				int device_pos;

			private:
				run_kernels_params() = delete;
				run_kernels_params(const run_kernels_params&) = delete;
				run_kernels_params& operator =(const run_kernels_params&) = delete;
			};

			static void run_kernels_static(forward_propagation_cuda * self, std::shared_ptr<run_kernels_params> params);
			void run_kernels(run_kernels_params& params);

			unsigned int run_kernels_thread_io_set;

			bool interrupt_thread;

		private:
			class read_entry_info
			{
			public:
				typedef std::shared_ptr<read_entry_info> ptr;

				read_entry_info() = default;

				unsigned int entry_id;
				std::map<std::string, float *> data_map;
				bool entry_read;

				structured_data_bunch_reader * reader;

				bool read_entry_finished;
				std::mutex read_entry_finished_mutex;
				std::condition_variable read_entry_finished_condition;
				std::string error_message;

			private:
				read_entry_info(const read_entry_info&) = delete;
				read_entry_info& operator =(const read_entry_info&) = delete;
			};

			static void read_input_data_static(read_entry_info * params);

		private:
			cuda_multi_running_configuration::const_ptr cuda_multi_config;
			int config_count;

			std::vector<network_action_schema::const_ptr> optimized_action_schema_list;
			std::vector<std::vector<layer_name_with_action>> actions_in_execution_order_list;
			std::vector<std::map<layer_name_with_action, std::pair<cuda_event::ptr, cuda_event::ptr>>> start_stop_profiling_events_list;

			network_data::const_ptr host_net_data;

			std::map<std::string, layer_testing_schema::const_ptr> testing_schemas;
			std::vector<std::map<std::string, layer_tester_cuda::ptr>> testers_list;

			std::vector<cuda_stream::ptr> copy_data_stream_list;

			std::vector<std::vector<cuda_stream::ptr>> command_streams_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> action_to_stream_set_map_list;
			std::vector<std::map<layer_name_with_action, cuda_event::ptr>> action_output_data_ready_events_list;
			std::vector<std::map<layer_name_with_action, std::vector<cuda_event::ptr>>> action_previous_events_list;
			std::vector<std::vector<cuda_event::ptr>> output_data_ready_additional_events_list;
			std::vector<unsigned int> output_data_ready_stream_set_id_list;

			std::vector<std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>> schema_data_list;
			std::vector<std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>> net_data_list;
			std::vector<std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>> net_data_custom_list;
			std::vector<std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>> persistent_working_data_list;

			std::vector<std::vector<size_t>> temporary_working_fixed_set_sizes_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> temporary_working_fixed_data_action_to_set_map_list;

			std::vector<std::vector<size_t>> layer_buffer_set_per_entry_sizes_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> temporary_working_per_entry_data_action_to_set_map_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> layer_buffer_action_to_set_map_list;

			std::map<std::string, size_t> dedicated_per_entry_data_name_to_size_map;

			std::map<std::string, size_t> input_per_entry_host_data_name_to_size_map;
			std::map<std::string, size_t> output_per_entry_host_data_name_to_size_map;

			std::vector<unsigned int> max_entry_count_list;

		private:
			static const unsigned int max_max_entry_count;

		private:
			forward_propagation_cuda(const forward_propagation_cuda&) = delete;
			forward_propagation_cuda& operator =(const forward_propagation_cuda&) = delete;
		};
	}
}
