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

#include "../backward_propagation.h"
#include "cuda_multi_running_configuration.h"
#include "layer_updater_schema.h"
#include "cuda_stream.h"
#include "cuda_event.h"

#include <map>
#include <condition_variable>
#include <mutex>

namespace nnforge
{
	namespace cuda
	{
		class backward_propagation_cuda : public backward_propagation
		{
		public:
			backward_propagation_cuda(
				const network_schema& schema,
				const std::vector<std::string>& output_layer_names,
				const std::vector<std::string>& error_source_layer_names,
				const std::vector<std::string>& exclude_data_update_layer_names,
				debug_state::ptr debug,
				profile_state::ptr profile,
				cuda_multi_running_configuration::const_ptr cuda_multi_config);

			virtual ~backward_propagation_cuda() = default;

		protected:
			// schema, network data and data are guaranteed to be compatible
			// The function should set average absolute updates, the number of entries processed, and optionally time it takes to run each action
			virtual void actual_run(
				structured_data_bunch_reader& reader,
				structured_data_bunch_writer& writer,
				network_data& data,
				network_data::ptr momentum_data,
				network_data::ptr momentum_data2,
				const std::map<std::string, std::vector<float> >& learning_rates,
				unsigned int batch_size,
				float weight_decay,
				training_momentum momentum,
				unsigned int epoch_id,
				std::map<std::string, std::vector<float> >& average_absolute_updates,
				unsigned int& entries_processed,
				std::map<layer_name_with_action, float>& action_seconds);

			// The method is called when client calls set_input_configuration_specific and the configuration is modified.
			// The layer_config_map is guaranteed to be compatible with schema
			virtual void layer_config_map_modified();

			virtual float get_max_flops() const;

		private:
			void setup_network_cuda();

			void setup_streams_and_events();

			void setup_io_host_buffer_sizes();

			void setup_dedicated_buffer_sizes();

			void setup_fixed_buffer_sizes();

			void setup_per_entry_buffer_sizes();

			void update_buffer_config();

			void setup_optimized_action_schema();

			std::vector<std::map<std::string, std::vector<cuda_linear_buffer_device::ptr>>> get_data(const layer_data_list& host_data) const;

			std::vector<std::map<std::string, std::vector<cuda_linear_buffer_device::ptr>>> get_zero_gradient(const std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& net_data) const;

			void read_data(
				const std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& data_list,
				layer_data_list& host_data) const;

			std::map<std::string, std::vector<float> > read_update_accum(
				const std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& update_accum_buffers,
				const std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& data,
				unsigned int gradient_applied_count) const;

			std::vector<std::vector<unsigned int>> get_entry_read_counts_list(
				const std::vector<unsigned int>& max_entry_count_list,
				unsigned int batch_size) const;

		void enqueue_apply_gradient(
			cudaStream_t stream_id,
			const std::string& layer_name,
			std::vector<cuda_linear_buffer_device::ptr>& data,
			std::vector<cuda_linear_buffer_device::ptr>& gradient,
			std::vector<cuda_linear_buffer_device::ptr>& prev_upd,
			std::vector<cuda_linear_buffer_device::ptr>& prev_upd2,
			const std::vector<float>& learning_rates,
			const std::vector<cuda_linear_buffer_device::ptr>& update_accum_buffers,
			float gradient_normalizer,
			float weight_decay,
			training_momentum momentum,
			unsigned int iteration_id,
			int device_pos);

		private:
			class run_kernels_params
			{
			public:
				run_kernels_params(
					int device_pos,
					std::map<std::string, std::array<cuda_linear_buffer_device::ptr, 2> >& dedicated_buffers,
					std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& net_data,
					std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >& net_data_custom,
					std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >& persistent_working_data,
					std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& gradient,
					std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& previous_upd,
					std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& previous_upd2,
					std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& update_accum_buffers,
					const std::map<std::string, std::vector<float> >& learning_rates,
					unsigned int batch_size,
					float weight_decay,
					training_momentum momentum,
					unsigned int max_chunk_size,
					unsigned int base_iteration_count,
					bool update_accum_stats);

				std::map<std::string, std::array<cuda_linear_buffer_device::ptr, 2> >& dedicated_buffers;
				std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& net_data;
				std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >& net_data_custom;
				std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >& persistent_working_data;
				std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& gradient;
				std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& previous_upd;
				std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& previous_upd2;
				std::map<std::string, std::vector<cuda_linear_buffer_device::ptr> >& update_accum_buffers;
				const std::map<std::string, std::vector<float> >& learning_rates;
				unsigned int batch_size;
				float weight_decay;
				training_momentum momentum;
				unsigned int max_chunk_size;
				unsigned int base_iteration_count;

				std::map<layer_name_with_action, double> action_seconds;

				unsigned int gradient_applied_count;

				std::string error_message;

				unsigned int run_kernels_thread_entry_to_process_count;

				bool run_kernels_finished;
				std::mutex run_kernels_finished_mutex;
				std::condition_variable run_kernels_finished_condition;

				bool run_kernels_task_ready;
				std::mutex run_kernels_pending_mutex;
				std::condition_variable run_kernels_pending_condition;

				int device_pos;

				float gradient_normalizer;
				bool apply_gradient;
				bool final_iteration;
				bool update_accum_stats;

			private:
				run_kernels_params() = delete;
				run_kernels_params(const run_kernels_params&) = delete;
				run_kernels_params& operator =(const run_kernels_params&) = delete;
			};

			static void run_kernels_static(backward_propagation_cuda * self, std::shared_ptr<run_kernels_params> params);
			void run_kernels(run_kernels_params& params);

			unsigned int run_kernels_thread_io_set;

			bool interrupt_thread;

		private:
			class read_entry_info
			{
			public:
				typedef std::shared_ptr<read_entry_info> ptr;

				read_entry_info();

				unsigned int entry_id;
				std::map<std::string, float *> data_map;
				bool entry_read;

				structured_data_bunch_reader * reader;

				bool read_entry_finished;
				std::mutex read_entry_finished_mutex;
				std::condition_variable read_entry_finished_condition;
				std::string error_message;

			private:
				read_entry_info(const read_entry_info&);
				read_entry_info& operator =(const read_entry_info&);
			};

			static void read_input_data_static(read_entry_info * params);

		private:
			cuda_multi_running_configuration::const_ptr cuda_multi_config;
			int config_count;

			std::vector<network_action_schema::const_ptr> optimized_action_schema_list;
			std::vector<std::vector<layer_name_with_action>> actions_in_execution_order_list;
			std::vector<std::map<layer_name_with_action, std::pair<cuda_event::ptr, cuda_event::ptr>>> start_stop_profiling_events_list;
			std::map<std::string, std::vector<layer_name_with_action> > input_to_all_output_map;

			std::map<std::string, layer_updater_schema::const_ptr> updater_schemas;
			std::vector<std::map<std::string, layer_updater_cuda::ptr>> updaters_list;

			std::vector<cuda_stream::ptr> copy_data_stream_list;

			std::vector<std::vector<cuda_stream::ptr>> command_streams_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> action_to_stream_set_map_list;
			std::vector<std::map<layer_name_with_action, cuda_event::ptr>> action_output_data_ready_events_list;
			std::vector<std::map<layer_name_with_action, std::vector<cuda_event::ptr>>> action_previous_events_list;
			std::vector<std::vector<cuda_event::ptr>> output_data_ready_additional_events_list;
			std::vector<unsigned int> output_data_ready_stream_set_id_list;

			std::vector<std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>> schema_data_list;

			std::vector<std::vector<size_t>> fixed_set_sizes_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> temporary_working_fixed_data_action_to_set_map_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> temporary_fixed_data_action_to_set_map_list;

			std::vector<std::vector<size_t>> layer_buffer_set_per_entry_sizes_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> temporary_working_per_entry_data_action_to_set_map_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> layer_buffer_action_to_set_map_list;
			std::vector<std::map<layer_name_with_action, unsigned int>> temporary_per_entry_data_action_to_set_map_list;

			std::map<std::string, size_t> dedicated_per_entry_data_name_to_size_map;

			std::map<std::string, size_t> input_per_entry_host_data_name_to_size_map;
			std::map<std::string, size_t> output_per_entry_host_data_name_to_size_map;

			std::vector<buffer_cuda_size_configuration> buffer_config_without_data_and_momentum_list;

		private:
			static const unsigned int elem_count_update_accum_per_part;
			static bool dump_data;

		private:
			backward_propagation_cuda(const backward_propagation_cuda&) = delete;
			backward_propagation_cuda& operator =(const backward_propagation_cuda&) = delete;
		};
	}
}
