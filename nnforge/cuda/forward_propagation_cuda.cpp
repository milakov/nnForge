/*
 *  Copyright 2011-2017 Maxim Milakov
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

#include "forward_propagation_cuda.h"

#include "layer_testing_schema_factory.h"
#include "cuda_linear_buffer_host.h"
#include "neural_network_cuda_exception.h"
#include "cuda_profiling.h"

#include "../data_layer.h"
#include "../neural_network_exception.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <thread>
#include <functional>
#include <numeric>
#include <chrono>

namespace nnforge
{
	namespace cuda
	{
		const unsigned int forward_propagation_cuda::max_max_entry_count = 1024;

		forward_propagation_cuda::forward_propagation_cuda(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			debug_state::ptr debug,
			profile_state::ptr profile,
			cuda_multi_running_configuration::const_ptr cuda_multi_config)
			: forward_propagation(schema, output_layer_names, debug, profile)
			, cuda_multi_config(cuda_multi_config)
			, config_count(static_cast<int>(cuda_multi_config->cuda_config_list.size()))
			, max_entry_count_list(config_count, 0)
		{
			setup_network_cuda();

			std::vector<layer_name_with_action> actions = action_schema->get_actions();

			for(std::vector<layer_name_with_action>::const_iterator it = actions.begin(); it != actions.end(); ++it)
				testing_schemas.insert(
					std::make_pair(
						it->get_name(),
						layer_testing_schema_factory::get_singleton().create_testing_schema_layer(this->schema->get_layer(it->get_name()))));

			for(auto cuda_config: cuda_multi_config->cuda_config_list)
			{
				cuda_config->set_device();

				schema_data_list.push_back(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>());
				auto& schema_data = schema_data_list.back();
				for(std::map<std::string, layer_testing_schema::const_ptr>::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it)
					schema_data.insert(std::make_pair(it->first, it->second->get_schema_buffers()));
			}
		}

		void forward_propagation_cuda::actual_run(
			structured_data_bunch_reader& reader,
			structured_data_bunch_writer& writer,
			unsigned int& entries_processed,
			std::map<layer_name_with_action, float>& action_seconds,
			float& idle_seconds)
		{
			std::vector<unsigned int> current_max_entry_count_list = max_entry_count_list;
			int reader_entry_count = reader.get_entry_count();
			if (reader_entry_count > 0)
				std::for_each(current_max_entry_count_list.begin(), current_max_entry_count_list.end(), [reader_entry_count] (unsigned int& x) { x = std::min(x, static_cast<unsigned int>(reader_entry_count)); });

			std::vector<std::map<std::string, std::array<cuda_linear_buffer_device::ptr, 2>>> dedicated_buffers_list;
			std::vector<std::map<std::string, cuda_linear_buffer_host::ptr>> input_host_buffers_list;
			std::vector<std::map<std::string, cuda_linear_buffer_host::ptr>> output_host_buffers_list;
			for(int i = 0; i < config_count; ++i)
			{
				cuda_multi_config->cuda_config_list[i]->set_device();

				dedicated_buffers_list.push_back(std::map<std::string, std::array<cuda_linear_buffer_device::ptr, 2>>());
				auto& dedicated_buffers = dedicated_buffers_list.back();
				input_host_buffers_list.push_back(std::map<std::string, cuda_linear_buffer_host::ptr>());
				auto& input_host_buffers = input_host_buffers_list.back();
				output_host_buffers_list.push_back(std::map<std::string, cuda_linear_buffer_host::ptr>());
				auto& output_host_buffers = output_host_buffers_list.back();

				for(auto it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
				{
					std::array<cuda_linear_buffer_device::ptr, 2>& arr = dedicated_buffers.insert(std::make_pair(it->first, std::array<cuda_linear_buffer_device::ptr, 2>())).first->second;
					arr[0] = cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(it->second * current_max_entry_count_list[i]));
					arr[1] = cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(it->second * current_max_entry_count_list[i]));
				}

				for(auto it = input_per_entry_host_data_name_to_size_map.begin(); it != input_per_entry_host_data_name_to_size_map.end(); ++it)
					input_host_buffers.insert(std::make_pair(it->first,
						cuda_linear_buffer_host::ptr(new cuda_linear_buffer_host(it->second * current_max_entry_count_list[i]))));

				for(auto it = output_per_entry_host_data_name_to_size_map.begin(); it != output_per_entry_host_data_name_to_size_map.end(); ++it)
					output_host_buffers.insert(std::make_pair(it->first,
						cuda_linear_buffer_host::ptr(new cuda_linear_buffer_host(it->second * current_max_entry_count_list[i]))));
			}
	
			unsigned int entry_processed_count = 0;

			std::vector<std::shared_ptr<run_kernels_params>> params_list;
			for(int i = 0; i < config_count; ++i)
				params_list.push_back(std::shared_ptr<run_kernels_params>(new run_kernels_params(i, dedicated_buffers_list[i], current_max_entry_count_list[i])));
			interrupt_thread = false;
			std::vector<std::thread> run_kernels_thread_list;
			for(int i = 0; i < config_count; ++i)
				run_kernels_thread_list.push_back(std::thread(run_kernels_static, this, params_list[i]));
			try
			{
				run_kernels_thread_io_set = 0;
				bool initial_iteration = true;
				bool try_to_read = true;
				std::vector<bool> run_kernels_thread_stopped_list(config_count, false);
				bool entry_not_read_encountered = false;
				std::vector<unsigned int> entry_to_process_count_list(config_count, 0);
				std::vector<unsigned int> entry_to_write_count_list(config_count, 0);
				unsigned int base_entry_to_read_id = 0;
				std::vector<std::vector<read_entry_info::ptr>> read_entry_info_list_list;
				for(int device_pos = 0; device_pos < config_count; ++device_pos)
				{
					read_entry_info_list_list.push_back(std::vector<read_entry_info::ptr>(current_max_entry_count_list[device_pos]));
					auto& read_entry_info_list = read_entry_info_list_list.back();
					for(unsigned int i = 0; i < current_max_entry_count_list[device_pos]; ++i)
					{
						read_entry_info_list[i] = read_entry_info::ptr(new read_entry_info());
						read_entry_info_list[i]->reader = &reader;
						for(auto it = input_per_entry_host_data_name_to_size_map.begin(); it != input_per_entry_host_data_name_to_size_map.end(); ++it)
							read_entry_info_list[i]->data_map.insert(std::make_pair(it->first, (float *)(*input_host_buffers_list[device_pos][it->first]) + i * (it->second / sizeof(float))));
					}
				}

				while(true)
				{
					unsigned int copy_data_thread_io_set = 1 - run_kernels_thread_io_set;
					std::vector<bool> wait_for_kernels_to_finish_list(config_count, false);
					if (!initial_iteration)
					{
						// Set command
						for(int i = 0; i < config_count; ++i)
						{
							if (!run_kernels_thread_stopped_list[i])
							{
								params_list[i]->run_kernels_thread_entry_to_process_count = entry_to_process_count_list[i];
								params_list[i]->run_kernels_finished = false;
								{
									std::lock_guard<std::mutex> lock(params_list[i]->run_kernels_pending_mutex);
									params_list[i]->run_kernels_task_ready = true;
								}
								params_list[i]->run_kernels_pending_condition.notify_one();
								run_kernels_thread_stopped_list[i] = (params_list[i]->run_kernels_thread_entry_to_process_count == 0);
								wait_for_kernels_to_finish_list[i] = !run_kernels_thread_stopped_list[i];
							}
						}
					}

					// Launch D2H copy for output data
					for(int i = 0; i < config_count; ++i)
					{
						if (entry_to_write_count_list[i] > 0)
						{
							cuda_multi_config->cuda_config_list[i]->set_device();
							for(auto it = output_host_buffers_list[i].begin(); it != output_host_buffers_list[i].end(); ++it)
							{
								cuda_safe_call(cudaMemcpyAsync(
									*it->second,
									*dedicated_buffers_list[i][it->first][copy_data_thread_io_set],
									output_per_entry_host_data_name_to_size_map[it->first] * entry_to_write_count_list[i],
									cudaMemcpyDeviceToHost,
									*copy_data_stream_list[i]));
							}
							if (cuda_multi_config->cuda_config_list[i]->is_flush_required())
								cuda_relaxed_safe_call(cudaStreamQuery(*copy_data_stream_list[i]));
						}
					}

					std::vector<unsigned int> entry_read_count_list(config_count, 0);
					if (!entry_not_read_encountered)
					{
						PUSH_RANGE("Launching reading input data", 0);
						// Launch all read input data tasks
						for(int device_pos = 0; device_pos < config_count; ++device_pos)
						{
							for(unsigned int i = 0; i < current_max_entry_count_list[device_pos]; ++i)
							{
								read_entry_info& current_info = *read_entry_info_list_list[device_pos][i];
								current_info.read_entry_finished = false;
								current_info.entry_id = base_entry_to_read_id;
								cuda_multi_config->get_job_runner()->service.post(std::bind(read_input_data_static, &current_info));
								++base_entry_to_read_id;
							}
						}
						POP_RANGE;
					}

					std::string read_input_error_message;
					for(int device_pos = 0; device_pos < config_count; ++device_pos)
					{
						cuda_multi_config->cuda_config_list[device_pos]->set_device();

						// Wait for input data to be read
						if (!entry_not_read_encountered)
						{
							std::string profiling_str = (boost::format("Waiting input data for device %1%") % cuda_multi_config->cuda_config_list[device_pos]->device_id).str();
							PUSH_RANGE(profiling_str.c_str(), 3);
							for(unsigned int i = 0; i < current_max_entry_count_list[device_pos]; ++i)
							{
								read_entry_info& current_info = *read_entry_info_list_list[device_pos][i];

								{
									std::unique_lock<std::mutex> lock(current_info.read_entry_finished_mutex);
									while (!current_info.read_entry_finished)
										current_info.read_entry_finished_condition.wait(lock);
								}
								if (!current_info.error_message.empty() && read_input_error_message.empty())
								{
									read_input_error_message = current_info.error_message;
								}
								if (!entry_not_read_encountered)
								{
									if (current_info.entry_read)
										++entry_read_count_list[device_pos];
									else
										entry_not_read_encountered = true;
								}
							}
							POP_RANGE;
						}

						// Make sure output data is copied to host
						std::string profiling_str = (boost::format("Waiting for output data transfer for device %1%") % cuda_multi_config->cuda_config_list[device_pos]->device_id).str();
						PUSH_RANGE(profiling_str.c_str(), 2);
						cuda_safe_call(cudaStreamSynchronize(*copy_data_stream_list[device_pos]));
						POP_RANGE;

						// Launch H2D copy for input data
						if (entry_read_count_list[device_pos] > 0)
						{
							for(auto it = input_host_buffers_list[device_pos].begin(); it != input_host_buffers_list[device_pos].end(); ++it)
							{
								cuda_safe_call(cudaMemcpyAsync(
									*dedicated_buffers_list[device_pos][it->first][copy_data_thread_io_set],
									*it->second,
									input_per_entry_host_data_name_to_size_map[it->first] * entry_read_count_list[device_pos],
									cudaMemcpyDeviceToHost,
									*copy_data_stream_list[device_pos]));
							}
							if (cuda_multi_config->cuda_config_list[device_pos]->is_flush_required())
								cuda_relaxed_safe_call(cudaStreamQuery(*copy_data_stream_list[device_pos]));
						}
	
						// Write output data
						if (entry_to_write_count_list[device_pos] > 0)
						{
							std::string profiling_str = (boost::format("Writing output data for device %1%") % cuda_multi_config->cuda_config_list[device_pos]->device_id).str();
							PUSH_RANGE(profiling_str.c_str(), 1);
							for(unsigned int i = 0; i < entry_to_write_count_list[device_pos] * output_layers_tiling_factor; ++i)
							{
								std::map<std::string, const float *> data_map;
								for(std::map<std::string, size_t>::const_iterator it = output_per_entry_host_data_name_to_size_map.begin(); it != output_per_entry_host_data_name_to_size_map.end(); ++it)
									data_map.insert(std::make_pair(it->first, (float *)(*output_host_buffers_list[device_pos][it->first]) + i * (it->second / sizeof(float) / output_layers_tiling_factor)));
								writer.write(entry_processed_count, data_map);
								++entry_processed_count;
							}
							POP_RANGE;
						}
					} // for(int device_pos
					if (!read_input_error_message.empty())
						throw neural_network_exception(read_input_error_message);

					// Make sure input data is copied to device
					PUSH_RANGE("Waiting for input data to be transfered", 4);
					for(int device_pos = 0; device_pos < config_count; ++device_pos)
						cuda_safe_call(cudaStreamSynchronize(*copy_data_stream_list[device_pos]));
					POP_RANGE;

					for(int device_pos = 0; device_pos < config_count; ++device_pos)
					{
						if (wait_for_kernels_to_finish_list[device_pos])
						{
							PUSH_RANGE("Waiting for kernels to finish", 2);
							// Wait for all the kernels to finish execution
							{
								std::unique_lock<std::mutex> lock(params_list[device_pos]->run_kernels_finished_mutex);
								while (!params_list[device_pos]->run_kernels_finished)
									params_list[device_pos]->run_kernels_finished_condition.wait(lock);
							}
							POP_RANGE;
							if (!params_list[device_pos]->error_message.empty())
								throw neural_network_exception(params_list[device_pos]->error_message);
						}
					}

					run_kernels_thread_io_set = 1 - run_kernels_thread_io_set; // Switch set of IO buffers
					initial_iteration = false;
					unsigned int current_entry_read_count = std::accumulate(entry_read_count_list.begin(), entry_read_count_list.end(), 0);
					entry_to_write_count_list = entry_to_process_count_list;
					entry_to_process_count_list = entry_read_count_list;

					if ((current_entry_read_count == 0) && (!std::any_of(wait_for_kernels_to_finish_list.begin(), wait_for_kernels_to_finish_list.end(), [] (bool x) { return x; })))
						break;
				}
			}
			catch (const std::exception&)
			{
				interrupt_thread = true;
				for(auto& t: run_kernels_thread_list)
					t.join();
				throw;
			}

			for(auto& t: run_kernels_thread_list)
				t.join();
			for(auto p: params_list)
				if (!p->error_message.empty())
					throw neural_network_exception(p->error_message);

			entries_processed = entry_processed_count;

			action_seconds.clear();
			idle_seconds = 0.0F;
			float mult = 1.0F / static_cast<float>(config_count);
			for(auto& p: params_list)
			{
				idle_seconds += static_cast<float>(p->idle_seconds) * mult;
				for(auto dt: p->action_seconds)
				{
					auto it = action_seconds.find(dt.first);
					if (it == action_seconds.end())
						it = action_seconds.insert(std::make_pair(dt.first, 0.0F)).first;
					it->second += static_cast<float>(dt.second) * mult;
				}
			}
		}

		void forward_propagation_cuda::read_input_data_static(read_entry_info * params)
		{
			try
			{
				params->entry_read = params->reader->read(params->entry_id, params->data_map);

				// Notify caller thread that result is ready
				{
					std::lock_guard<std::mutex> lock(params->read_entry_finished_mutex);
					params->read_entry_finished = true;
				}
				params->read_entry_finished_condition.notify_one();
			}
			catch (const std::runtime_error& e)
			{
				params->error_message = e.what();
				{
					std::lock_guard<std::mutex> lock(params->read_entry_finished_mutex);
					params->read_entry_finished = true;
				}
				params->read_entry_finished_condition.notify_one();
			}
		}

		void forward_propagation_cuda::run_kernels(run_kernels_params& params)
		{
			try
			{
				cuda_multi_config->cuda_config_list[params.device_pos]->set_device();

				std::vector<cuda_linear_buffer_device::ptr> temporary_working_fixed_buffers;
				for(auto sz: temporary_working_fixed_set_sizes_list[params.device_pos])
					temporary_working_fixed_buffers.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(sz)));

				std::vector<cuda_linear_buffer_device::ptr> layer_buffers;
				for(auto sz: layer_buffer_set_per_entry_sizes_list[params.device_pos])
					layer_buffers.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(sz * params.current_max_entry_count)));

				std::unique_lock<std::mutex> lock(params.run_kernels_pending_mutex);
				while(true)
				{
					if (interrupt_thread)
						break;

					std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
					while (!params.run_kernels_task_ready)
						params.run_kernels_pending_condition.wait(lock);
					std::chrono::duration<double> idle_sec = std::chrono::high_resolution_clock::now() - start;
					params.idle_seconds += idle_sec.count();

					params.run_kernels_task_ready = false;

					if (params.run_kernels_thread_entry_to_process_count == 0)
						break;

					std::set<layer_name_with_action> actions_profiled;
					for(const auto& current_layer_name_with_action: actions_in_execution_order_list[params.device_pos])
					{
						std::string layer_name = current_layer_name_with_action.get_name();;
						layer_action action = current_layer_name_with_action.get_action();
						layer::const_ptr current_layer = schema->find_layer(layer_name);

						cuda_stream::ptr current_stream = command_streams_list[params.device_pos][action_to_stream_set_map_list[params.device_pos][current_layer_name_with_action]];

						// Enqueue waits for previous events
						{
							auto previous_events_it = action_previous_events_list[params.device_pos].find(current_layer_name_with_action);
							if (previous_events_it != action_previous_events_list[params.device_pos].end())
							{
								const std::vector<cuda_event::ptr>& previous_events = previous_events_it->second;
								for(std::vector<cuda_event::ptr>::const_iterator event_it = previous_events.begin(); event_it != previous_events.end(); ++event_it)
									cuda_safe_call(cudaStreamWaitEvent(*current_stream, **event_it, 0));
							}
						}

						// Enqueue forward prop
						{
							cuda_linear_buffer_device::ptr output_buffer;
							{
								auto it = layer_buffer_action_to_set_map_list[params.device_pos].find(current_layer_name_with_action);
								if (it != layer_buffer_action_to_set_map_list[params.device_pos].end())
									output_buffer = layer_buffers[it->second];
								else
									output_buffer = params.dedicated_buffers.find(layer_name)->second[run_kernels_thread_io_set];
							}

							std::vector<cuda_linear_buffer_device::const_ptr> input_buffers;
							for(const auto& in: current_layer->input_layer_instance_names)
							{
								std::map<layer_name_with_action, unsigned int>::const_iterator it = layer_buffer_action_to_set_map_list[params.device_pos].find(layer_name_with_action(in, layer_action::forward));
								if (it != layer_buffer_action_to_set_map_list[params.device_pos].end())
									input_buffers.push_back(layer_buffers[it->second]);
								else
									input_buffers.push_back(params.dedicated_buffers.find(in)->second[run_kernels_thread_io_set]);
							}

							cuda_linear_buffer_device::ptr temporary_working_fixed_buffer;
							{
								auto it = temporary_working_fixed_data_action_to_set_map_list[params.device_pos].find(current_layer_name_with_action);
								if (it != temporary_working_fixed_data_action_to_set_map_list[params.device_pos].end())
									temporary_working_fixed_buffer = temporary_working_fixed_buffers[it->second];
							}

							cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer;
							{
								auto it = temporary_working_per_entry_data_action_to_set_map_list[params.device_pos].find(current_layer_name_with_action);
								if (it != temporary_working_per_entry_data_action_to_set_map_list[params.device_pos].end())
									temporary_working_per_entry_buffer = layer_buffers[it->second];
							}

							std::vector<cuda_linear_buffer_device::const_ptr> data_list;
							{
								auto data_list_it = net_data_list[params.device_pos].find(layer_name);
								if (data_list_it != net_data_list[params.device_pos].end())
									data_list = data_list_it->second;
							}

							std::vector<cuda_linear_buffer_device::const_ptr> data_custom_list;
							{
								auto data_custom_list_it = net_data_custom_list[params.device_pos].find(layer_name);
								if (data_custom_list_it != net_data_custom_list[params.device_pos].end())
									data_custom_list = data_custom_list_it->second;
							}

							if (profile->is_profile())
							{
								cuda_safe_call(cudaEventRecord(*start_stop_profiling_events_list[params.device_pos][current_layer_name_with_action].first, *current_stream));
								actions_profiled.insert(current_layer_name_with_action);
							}

							testers_list[params.device_pos].find(layer_name)->second->enqueue_forward_propagation(
								*current_stream,
								output_buffer,
								schema_data_list[params.device_pos][layer_name],
								data_list,
								data_custom_list,
								input_buffers,
								persistent_working_data_list[params.device_pos][layer_name],
								temporary_working_fixed_buffer,
								temporary_working_per_entry_buffer,
								params.run_kernels_thread_entry_to_process_count * cumulative_tiling_factor_map[layer_name]);

							if (profile->is_profile())
								cuda_safe_call(cudaEventRecord(*start_stop_profiling_events_list[params.device_pos][current_layer_name_with_action].second, *current_stream));
						}

						// Enqeue event
						{
							auto current_event_it = action_output_data_ready_events_list[params.device_pos].find(current_layer_name_with_action);
							if (current_event_it != action_output_data_ready_events_list[params.device_pos].end())
								cuda_safe_call(cudaEventRecord(*current_event_it->second, *current_stream));
						}

						if (cuda_multi_config->cuda_config_list[params.device_pos]->is_flush_required())
							cuda_relaxed_safe_call(cudaStreamQuery(*current_stream));
					}

					// Wait for output data to be ready
					for(auto event: output_data_ready_additional_events_list[params.device_pos])
						cuda_safe_call(cudaStreamWaitEvent(*command_streams_list[params.device_pos][output_data_ready_stream_set_id_list[params.device_pos]], *event, 0));

					// Wait for all kernels to finish
					cuda_safe_call(cudaStreamSynchronize(*command_streams_list[params.device_pos][output_data_ready_stream_set_id_list[params.device_pos]]));
		
					if (profile->is_profile())
					{
						for(const auto& ac: actions_profiled)
						{
							auto it = start_stop_profiling_events_list[params.device_pos].find(ac);
							float milliseconds;
							cuda_safe_call(cudaEventElapsedTime(&milliseconds, *it->second.first, *it->second.second));
							params.action_seconds.insert(std::make_pair(it->first, 0.0)).first->second += static_cast<double>(milliseconds * 0.001F);
						}
					}

					// Notify caller thread that result is ready
					{
						std::lock_guard<std::mutex> lock(params.run_kernels_finished_mutex);
						params.run_kernels_finished = true;
					}
					params.run_kernels_finished_condition.notify_one();
				}
			}
			catch (const std::runtime_error& e)
			{
				params.error_message = e.what();
				{
					std::lock_guard<std::mutex> lock(params.run_kernels_finished_mutex);
					params.run_kernels_finished = true;
				}
				params.run_kernels_finished_condition.notify_one();
			}
		}

		void forward_propagation_cuda::run_kernels_static(forward_propagation_cuda * self, std::shared_ptr<run_kernels_params> params)
		{
			self->run_kernels(*params);
		}

		forward_propagation_cuda::run_kernels_params::run_kernels_params(
			int device_pos,
			std::map<std::string, std::array<cuda_linear_buffer_device::ptr, 2> >& dedicated_buffers,
			unsigned int current_max_entry_count)
			: dedicated_buffers(dedicated_buffers)
			, current_max_entry_count(current_max_entry_count)
			, idle_seconds(0)
			, run_kernels_task_ready(false)
			, device_pos(device_pos)
		{
		}

		void forward_propagation_cuda::setup_network_cuda()
		{
			copy_data_stream_list.clear();
			for(auto cuda_config: cuda_multi_config->cuda_config_list)
			{
				cuda_config->set_device();
				copy_data_stream_list.push_back(cuda_stream::ptr(new cuda_stream()));
			}
		}

		void forward_propagation_cuda::setup_streams_and_events()
		{
			command_streams_list.clear();
			action_to_stream_set_map_list.clear();
			action_output_data_ready_events_list.clear();
			action_previous_events_list.clear();
			output_data_ready_additional_events_list.clear();
			start_stop_profiling_events_list.clear();
			output_data_ready_stream_set_id_list.clear();

			command_streams_list.resize(config_count);
			action_to_stream_set_map_list.resize(config_count);
			action_output_data_ready_events_list.resize(config_count);
			action_previous_events_list.resize(config_count);
			output_data_ready_additional_events_list.resize(config_count);
			start_stop_profiling_events_list.resize(config_count);
			output_data_ready_stream_set_id_list.resize(config_count);

			for(int i = 0; i < config_count; ++i)
			{
				auto cuda_config = cuda_multi_config->cuda_config_list[i];
				cuda_config->set_device();

				std::vector<std::vector<layer_name_with_action> > layer_stream_set = optimized_action_schema_list[i]->get_action_stream_set();

				if (cuda_multi_config->is_single_command_stream())
				{
					std::vector<std::vector<layer_name_with_action> > layer_stream_set_orig = layer_stream_set;
					layer_stream_set.clear();
					layer_stream_set.push_back(std::vector<layer_name_with_action>());
					std::vector<layer_name_with_action>& new_layer_list = layer_stream_set.front();
					for(std::vector<std::vector<layer_name_with_action> >::const_iterator it = layer_stream_set_orig.begin(); it != layer_stream_set_orig.end(); ++it)
					{
						const std::vector<layer_name_with_action>& ll = *it;
						for(std::vector<layer_name_with_action>::const_iterator it2 = ll.begin(); it2 != ll.end(); ++it2)
							new_layer_list.push_back(*it2);
					}
				}

				command_streams_list[i].resize(layer_stream_set.size());
				for(unsigned int stream_set_id = 0; stream_set_id < static_cast<unsigned int>(layer_stream_set.size()); ++stream_set_id)
				{
					command_streams_list[i][stream_set_id] = cuda_stream::ptr(new cuda_stream());
					for(std::vector<layer_name_with_action>::const_iterator it = layer_stream_set[stream_set_id].begin(); it != layer_stream_set[stream_set_id].end(); ++it)
						action_to_stream_set_map_list[i].insert(std::make_pair(*it, stream_set_id));
				}
				if (debug->is_debug())
				{
					debug->output_message((boost::format("forward prop cuda streams, device #%1% %2%: %3%") % cuda_config->device_id % cuda_config->device_name % layer_stream_set.size()).str().c_str());
					boost::filesystem::ofstream out(debug->get_path_to_unique_file((boost::format("forward_prop_cuda_streams_device_%1%") % cuda_config->device_id).str().c_str(), "gv"), std::ios_base::out | std::ios_base::trunc);
					optimized_action_schema_list[i]->write_gv(out, action_to_stream_set_map_list[i]);
				}

				for(auto it = actions_in_execution_order_list[i].rbegin(); it != actions_in_execution_order_list[i].rend(); ++it)
				{
					unsigned int current_stream_set_id = action_to_stream_set_map_list[i].find(*it)->second;

					std::vector<cuda_event::ptr> previous_events;
					std::vector<layer_name_with_action> previous_actions = optimized_action_schema_list[i]->get_dependencies(*it);
					for(std::vector<layer_name_with_action>::const_iterator it2 = previous_actions.begin(); it2 != previous_actions.end(); ++it2)
					{
						const layer_name_with_action& previous_layer_action = *it2;

						unsigned int previous_stream_set_id = action_to_stream_set_map_list[i].find(previous_layer_action)->second;
						if (previous_stream_set_id == current_stream_set_id)
							continue;

						cuda_event::ptr previous_event;
						auto it3 = action_output_data_ready_events_list[i].find(previous_layer_action);
						if (it3 != action_output_data_ready_events_list[i].end())
							previous_event = it3->second;
						else
							previous_event = action_output_data_ready_events_list[i].insert(std::make_pair(previous_layer_action, cuda_event::ptr(new cuda_event()))).first->second;
						previous_events.push_back(previous_event);
					}

					if (!previous_events.empty())
						action_previous_events_list[i].insert(std::make_pair(*it, previous_events));
				}

				bool output_data_ready_stream_set_id_defined = false;
				for(auto it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
				{
					if (!output_data_ready_stream_set_id_defined)
					{
						output_data_ready_stream_set_id_list[i] = action_to_stream_set_map_list[i][layer_name_with_action(*it, layer_action::forward)];
						output_data_ready_stream_set_id_defined = true;
						continue;
					}
					else
					{
						if (action_to_stream_set_map_list[i].find(layer_name_with_action(*it, layer_action::forward))->second == output_data_ready_stream_set_id_list[i])
							continue;
					}

					cuda_event::ptr previous_event;
					auto it3 = action_output_data_ready_events_list[i].find(layer_name_with_action(*it, layer_action::forward));
					if (it3 != action_output_data_ready_events_list[i].end())
						previous_event = it3->second;
					else
						previous_event = action_output_data_ready_events_list[i].insert(std::make_pair(layer_name_with_action(*it, layer_action::forward), cuda_event::ptr(new cuda_event()))).first->second;
					output_data_ready_additional_events_list[i].push_back(previous_event);
				}

				if (profile->is_profile())
				{
					for(auto it = actions_in_execution_order_list[i].begin(); it != actions_in_execution_order_list[i].end(); ++it)
						start_stop_profiling_events_list[i].insert(std::make_pair(*it, std::make_pair(cuda_event::ptr(new cuda_event(true)), cuda_event::ptr(new cuda_event(true)))));
				}
			}
		}

		void forward_propagation_cuda::actual_set_data(network_data::const_ptr data)
		{
			host_net_data = data;

			update_data();
		}

		void forward_propagation_cuda::actual_clear_data()
		{
			host_net_data.reset();

			update_data();
		}

		void forward_propagation_cuda::setup_optimized_action_schema()
		{
			optimized_action_schema_list.clear();
			actions_in_execution_order_list.clear();

			for(int i = 0; i < config_count; ++i)
			{
				auto cuda_config = cuda_multi_config->cuda_config_list[i];

				{
					network_action_schema::ptr optimized_action_schema_tmp = network_action_schema::ptr(new network_action_schema(*action_schema));
					float saturation_flops = cuda_config->get_flops() * cuda_config->get_device_saturation_time() / static_cast<float>(cuda_multi_config->optimize_action_graph_assumed_chunk_size);
					optimized_action_schema_tmp->add_dependencies_for_distant_otherwise_inependent_actions(
						layer_config_map,
						std::map<std::string, unsigned int>(),
						saturation_flops);
					optimized_action_schema_list.push_back(optimized_action_schema_tmp);
				}

				if (debug->is_debug())
				{
					boost::filesystem::ofstream out(debug->get_path_to_unique_file((boost::format("forward_prop_optimized_action_schema_device_%1%") % cuda_config->device_id).str().c_str(), "gv"), std::ios_base::out | std::ios_base::trunc);
					optimized_action_schema_list.back()->write_gv(out);
				}

				actions_in_execution_order_list.push_back(optimized_action_schema_list.back()->get_actions_in_execution_order());
			}
		}

		void forward_propagation_cuda::layer_config_map_modified()
		{
			setup_optimized_action_schema();

			setup_streams_and_events();

			testers_list.clear();

			setup_io_host_buffer_sizes();

			setup_dedicated_buffer_sizes();

			for(auto cuda_config: cuda_multi_config->cuda_config_list)
			{
				cuda_config->set_device();

				testers_list.push_back(std::map<std::string, layer_tester_cuda::ptr>());
				auto& testers = testers_list.back();
				for(std::map<std::string, layer_testing_schema::const_ptr>::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it)
				{
					const std::string& layer_name = it->first;
					layer_configuration_specific output_layer_configuration_specific = layer_config_map[layer_name];
					layer::const_ptr l = schema->get_layer(layer_name);
					std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
					for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
						input_layer_configuration_specific_list.push_back(layer_config_map[*it2]);

					testers.insert(
						std::make_pair(
							l->instance_name,
							it->second->create_tester(
								input_layer_configuration_specific_list,
								output_layer_configuration_specific,
								cuda_config)));
				}
			}

			setup_layer_buffer_sizes();

			setup_temporary_working_fixed_buffer_sizes();

			update_data();

			update_max_entry_count();
		}

		void forward_propagation_cuda::setup_layer_buffer_sizes()
		{
			layer_buffer_set_per_entry_sizes_list.clear();
			layer_buffer_action_to_set_map_list.clear();
			temporary_working_per_entry_data_action_to_set_map_list.clear();

			for(int i = 0; i < config_count; ++i)
			{
				auto cuda_config = cuda_multi_config->cuda_config_list[i];
				std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > layer_buffer_set_list;
				{
					std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > > buffers;
					std::map<layer_name_with_action, std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, bool> > > > dependencies;
					std::set<std::string> dedicated_output_buffers(output_layer_names.begin(), output_layer_names.end());
					for(auto it = actions_in_execution_order_list[i].begin(); it != actions_in_execution_order_list[i].end(); ++it)
					{
						std::string layer_name = it->get_name();
						int input_index_layer_can_write = testers_list[i][layer_name]->get_input_index_layer_can_write();
						size_t buffer_size_per_entry = layer_config_map.find(layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[layer_name] * sizeof(float);
						if (dedicated_output_buffers.find(layer_name) == dedicated_output_buffers.end())
							buffers.insert(std::make_pair(*it, std::vector<std::pair<buffer_lifetime, float> >(1, std::make_pair(buffer_lifetime(buffer_lifetime::action_output_buffer), static_cast<float>(buffer_size_per_entry)))));
						layer::const_ptr l = schema->get_layer(layer_name);
						std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, bool> > > current_dependencies;
						int input_index = 0;
						for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2, ++input_index)
						{
							const std::string& previous_layer_name = *it2;
							if (data_layer_names.find(previous_layer_name) == data_layer_names.end())
								current_dependencies.insert(std::make_pair(layer_name_with_action(previous_layer_name, layer_action(layer_action::forward)), std::vector<std::pair<buffer_lifetime, bool> >(1, std::make_pair(buffer_lifetime(buffer_lifetime::action_output_buffer), (input_index_layer_can_write == input_index)))));
						}
						if (!current_dependencies.empty())
							dependencies.insert(std::make_pair(*it, current_dependencies));
					}

					for(auto it = testers_list[i].begin(); it != testers_list[i].end(); ++it)
					{
						size_t temporary_working_per_entry_buffer_size = it->second->get_temporary_working_per_entry_buffer_size();
						if (temporary_working_per_entry_buffer_size > 0)
							buffers.insert(std::make_pair(layer_name_with_action(it->first, layer_action::forward), std::vector<std::pair<buffer_lifetime, float> >())).first->second.push_back(std::make_pair(buffer_lifetime(buffer_lifetime::working_buffer), static_cast<float>(temporary_working_per_entry_buffer_size)));
					}

					layer_buffer_set_list = optimized_action_schema_list[i]->get_buffer_set(
						buffers,
						dependencies,
						std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > >());

					if (cuda_multi_config->is_dont_share_buffers())
					{
						std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > layer_buffer_set_list_orig = layer_buffer_set_list;
						layer_buffer_set_list.clear();
						for(unsigned int set_id = 0; set_id < layer_buffer_set_list_orig.size(); ++set_id)
						{
							const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = layer_buffer_set_list_orig[set_id];
							for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
								layer_buffer_set_list.push_back(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >(1, *it));
						}
					}
				}

				layer_buffer_set_per_entry_sizes_list.push_back(std::vector<size_t>());
				auto& layer_buffer_set_per_entry_sizes = layer_buffer_set_per_entry_sizes_list.back();
				layer_buffer_action_to_set_map_list.push_back(std::map<layer_name_with_action, unsigned int>());
				auto& layer_buffer_action_to_set_map = layer_buffer_action_to_set_map_list.back();
				temporary_working_per_entry_data_action_to_set_map_list.push_back(std::map<layer_name_with_action, unsigned int>());
				auto& temporary_working_per_entry_data_action_to_set_map = temporary_working_per_entry_data_action_to_set_map_list.back();
				for(unsigned int set_id = 0; set_id < layer_buffer_set_list.size(); ++set_id)
				{
					const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = layer_buffer_set_list[set_id];
					size_t max_buffer_size_per_entry = 0;
					for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
					{
						std::string layer_name = it->first.get_name();
						size_t buffer_size_per_entry;
						if (it->second.get_buffer_lifetime_type() == buffer_lifetime::action_output_buffer)
						{
							layer_buffer_action_to_set_map.insert(std::make_pair(it->first, set_id));
							buffer_size_per_entry = layer_config_map.find(layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[layer_name] * sizeof(float);
						}
						else if (it->second.get_buffer_lifetime_type() == buffer_lifetime::working_buffer)
						{
							temporary_working_per_entry_data_action_to_set_map.insert(std::make_pair(it->first, set_id));
							buffer_size_per_entry = testers_list[i].find(layer_name)->second->get_temporary_working_per_entry_buffer_size() * cumulative_tiling_factor_map[layer_name];
						}
						else
							throw neural_network_exception((boost::format("Unexpected buffer lifetime %1% encountered for layer %2% action %3%") % it->second.str() % it->first.get_name() % it->first.get_action().str()).str());
						max_buffer_size_per_entry = std::max(max_buffer_size_per_entry, buffer_size_per_entry);
					}
					layer_buffer_set_per_entry_sizes.push_back(max_buffer_size_per_entry);
				}

				if (debug->is_debug())
				{
					std::stringstream debug_str;
					debug_str << "forward prop cuda per entry buffers, device #" << cuda_config->device_id << " " << cuda_config->device_name <<  ": " << layer_buffer_set_per_entry_sizes.size();
					size_t total_buffer_size = 0;
					for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_sizes.begin(); it != layer_buffer_set_per_entry_sizes.end(); ++it)
							total_buffer_size += *it;
					debug_str << ", total size " << ((total_buffer_size + 1024 - 1) / 1024) << " KB";
					debug->output_message(debug_str.str().c_str());
					for(unsigned int set_id = 0; set_id < static_cast<unsigned int>(layer_buffer_set_per_entry_sizes.size()); ++set_id)
					{
						std::stringstream debug_str;
						debug_str << " - " << ((layer_buffer_set_per_entry_sizes[set_id] + 1024 - 1) / 1024) << " KB: ";
						const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = layer_buffer_set_list[set_id];
						for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
						{
							if (it != action_list.begin())
								debug_str << ", ";
							debug_str << it->first.get_name();
							if (it->second.get_buffer_lifetime_type() != buffer_lifetime::action_output_buffer)
								debug_str << " " << it->second.str();
						}
						debug->output_message(debug_str.str().c_str());
					}
					boost::filesystem::ofstream out(debug->get_path_to_unique_file((boost::format("forward_prop_cuda_per_entry_buffers_device_%1%") % cuda_config->device_id).str().c_str(), "gv"), std::ios_base::out | std::ios_base::trunc);
					optimized_action_schema_list[i]->write_gv(out, layer_buffer_action_to_set_map, std::map<layer_name_with_action, unsigned int>(), temporary_working_per_entry_data_action_to_set_map);
				}
			}
		}

		void forward_propagation_cuda::setup_temporary_working_fixed_buffer_sizes()
		{
			temporary_working_fixed_set_sizes_list.clear();
			temporary_working_fixed_data_action_to_set_map_list.clear();

			for(int i = 0; i < config_count; ++i)
			{
				auto cuda_config = cuda_multi_config->cuda_config_list[i];
				size_t max_fixed_working_buffers_size = cuda_multi_config->cuda_config_list[i]->get_max_fixed_working_buffers_size();

				std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > temporary_working_fixed_buffer_set_list;
				{
					std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, float> > > buffers;
					for(auto it = testers_list[i].begin(); it != testers_list[i].end(); ++it)
					{
						std::pair<size_t, bool> temporary_working_fixed_buffer_size_and_flag = it->second->get_temporary_working_fixed_buffer_size();
						size_t temporary_working_fixed_buffer_size = temporary_working_fixed_buffer_size_and_flag.first;
						if (temporary_working_fixed_buffer_size_and_flag.second)
							temporary_working_fixed_buffer_size = std::max(temporary_working_fixed_buffer_size, max_fixed_working_buffers_size);
						if (temporary_working_fixed_buffer_size > 0)
							buffers.insert(std::make_pair(layer_name_with_action(it->first, layer_action::forward), std::vector<std::pair<buffer_lifetime, float> >())).first->second.push_back(std::make_pair(buffer_lifetime(buffer_lifetime::working_buffer), static_cast<float>(temporary_working_fixed_buffer_size)));
					}

					temporary_working_fixed_buffer_set_list = optimized_action_schema_list[i]->get_buffer_set(
						buffers,
						std::map<layer_name_with_action, std::map<layer_name_with_action, std::vector<std::pair<buffer_lifetime, bool> > > >(),
						std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > >());

					if (cuda_multi_config->is_dont_share_buffers())
					{
						std::vector<std::vector<std::pair<layer_name_with_action, buffer_lifetime> > > temporary_working_fixed_buffer_set_list_orig = temporary_working_fixed_buffer_set_list;
						temporary_working_fixed_buffer_set_list.clear();
						for(unsigned int set_id = 0; set_id < temporary_working_fixed_buffer_set_list_orig.size(); ++set_id)
						{
							const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = temporary_working_fixed_buffer_set_list_orig[set_id];
							for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
								temporary_working_fixed_buffer_set_list.push_back(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >(1, *it));
						}
					}
				}

				temporary_working_fixed_set_sizes_list.push_back(std::vector<size_t>());
				auto& temporary_working_fixed_set_size_list = temporary_working_fixed_set_sizes_list.back();
				temporary_working_fixed_data_action_to_set_map_list.push_back(std::map<layer_name_with_action, unsigned int>());
				auto& temporary_working_fixed_data_action_to_set_map = temporary_working_fixed_data_action_to_set_map_list.back();

				std::set<unsigned int> set_ids_with_hungry_working_buffers;
				for(unsigned int set_id = 0; set_id < temporary_working_fixed_buffer_set_list.size(); ++set_id)
				{
					const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = temporary_working_fixed_buffer_set_list[set_id];
					for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
					{
						std::string layer_name = it->first.get_name();
						if (testers_list[i].find(layer_name)->second->get_temporary_working_fixed_buffer_size().second)
							set_ids_with_hungry_working_buffers.insert(set_id);
					}
				}
				if (set_ids_with_hungry_working_buffers.size() > 1)
					max_fixed_working_buffers_size /= set_ids_with_hungry_working_buffers.size();

				for(unsigned int set_id = 0; set_id < temporary_working_fixed_buffer_set_list.size(); ++set_id)
				{
					const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = temporary_working_fixed_buffer_set_list[set_id];
					size_t max_buffer_size = (set_ids_with_hungry_working_buffers.find(set_id) != set_ids_with_hungry_working_buffers.end()) ? max_fixed_working_buffers_size : 1;
				 
					for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
					{
						std::string layer_name = it->first.get_name();
						temporary_working_fixed_data_action_to_set_map.insert(std::make_pair(it->first, set_id));
						size_t buffer_size = testers_list[i].find(layer_name)->second->get_temporary_working_fixed_buffer_size().first;
						max_buffer_size = std::max(max_buffer_size, buffer_size);
					}
					temporary_working_fixed_set_size_list.push_back(max_buffer_size);
				}

				if (debug->is_debug())
				{
					std::stringstream debug_str;
					debug_str << "forward prop cuda per fixed buffers, device #" << cuda_config->device_id << " " << cuda_config->device_name << ": " << temporary_working_fixed_set_size_list.size();
					size_t total_buffer_size = 0;
					for(std::vector<size_t>::const_iterator it = temporary_working_fixed_set_size_list.begin(); it != temporary_working_fixed_set_size_list.end(); ++it)
							total_buffer_size += *it;
					debug_str << ", total size " << ((total_buffer_size + (1024 * 1024) - 1) / (1024 * 1024)) << " MB";
					debug->output_message(debug_str.str().c_str());
					for(unsigned int set_id = 0; set_id < static_cast<unsigned int>(temporary_working_fixed_set_size_list.size()); ++set_id)
					{
						std::stringstream debug_str;
						debug_str << " - " << ((temporary_working_fixed_set_size_list[set_id] + (1024 * 1024) - 1) / (1024 * 1024)) << " MB: ";
						const std::vector<std::pair<layer_name_with_action, buffer_lifetime> >& action_list = temporary_working_fixed_buffer_set_list[set_id];
						for(std::vector<std::pair<layer_name_with_action, buffer_lifetime> >::const_iterator it = action_list.begin(); it != action_list.end(); ++it)
						{
							if (it != action_list.begin())
								debug_str << ", ";
							debug_str << it->first.get_name();
						}
						debug->output_message(debug_str.str().c_str());
					}
					boost::filesystem::ofstream out(debug->get_path_to_unique_file((boost::format("forward_prop_cuda_temporary_fixed_buffers_device_%1%") % cuda_config->device_id).str().c_str(), "gv"), std::ios_base::out | std::ios_base::trunc);
					optimized_action_schema_list[i]->write_gv(out, temporary_working_fixed_data_action_to_set_map);
				}
			}
		}

		void forward_propagation_cuda::setup_dedicated_buffer_sizes()
		{
			dedicated_per_entry_data_name_to_size_map.clear();

			std::set<std::string> separate_buffers_layer_names(output_layer_names.begin(), output_layer_names.end());
			separate_buffers_layer_names.insert(data_layer_names.begin(), data_layer_names.end());
			for(std::set<std::string>::const_iterator it = separate_buffers_layer_names.begin(); it != separate_buffers_layer_names.end(); ++it)
				dedicated_per_entry_data_name_to_size_map.insert(std::make_pair(*it, layer_config_map.find(*it)->second.get_neuron_count() * cumulative_tiling_factor_map[*it] * sizeof(float)));
		}

		void forward_propagation_cuda::setup_io_host_buffer_sizes()
		{
			input_per_entry_host_data_name_to_size_map.clear();
			output_per_entry_host_data_name_to_size_map.clear();

			for(std::set<std::string>::const_iterator it = data_layer_names.begin(); it != data_layer_names.end(); ++it)
				input_per_entry_host_data_name_to_size_map.insert(std::make_pair(*it, layer_config_map.find(*it)->second.get_neuron_count() * cumulative_tiling_factor_map[*it] * sizeof(float)));
			for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
				output_per_entry_host_data_name_to_size_map.insert(std::make_pair(*it, layer_config_map.find(*it)->second.get_neuron_count() * cumulative_tiling_factor_map[*it] * sizeof(float)));
		}

		void forward_propagation_cuda::update_data()
		{
			net_data_list.clear();
			net_data_custom_list.clear();
			persistent_working_data_list.clear();

			if (testers_list.empty() || (!host_net_data))
				return;

			for(int i = 0; i < config_count; ++i)
			{
				auto cuda_config = cuda_multi_config->cuda_config_list[i];
				cuda_config->set_device();

				net_data_list.push_back(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>());
				net_data_custom_list.push_back(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>());
				persistent_working_data_list.push_back(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr>>());

				auto& net_data = net_data_list.back();
				auto& net_data_custom = net_data_custom_list.back();
				auto& persistent_working_data = persistent_working_data_list.back();
				for(auto it = testers_list[i].begin(); it != testers_list[i].end(); ++it)
				{
					layer_data::const_ptr dt = host_net_data->data_list.find(it->first);
					if (dt)
						net_data.insert(std::make_pair(it->first, it->second->get_data(dt)));

					layer_data_custom::const_ptr dt_custom = host_net_data->data_custom_list.find(it->first);
					if (dt_custom)
						net_data_custom.insert(std::make_pair(it->first, it->second->set_get_data_custom(dt_custom)));

					persistent_working_data.insert(std::make_pair(it->first, it->second->get_persistent_working_data()));
				}
			}
		}

		void forward_propagation_cuda::update_max_entry_count()
		{
			for(int i = 0; i < config_count; ++i)
			{
				auto cuda_config = cuda_multi_config->cuda_config_list[i];

				buffer_cuda_size_configuration buffer_configuration;

				for(auto it = schema_data_list[i].begin(); it != schema_data_list[i].end(); ++it)
					for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
						buffer_configuration.add_constant_buffer((*it2)->get_size());
				for(auto it = net_data_list[i].begin(); it != net_data_list[i].end(); ++it)
					for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
						buffer_configuration.add_constant_buffer((*it2)->get_size());
				for(auto it = net_data_custom_list[i].begin(); it != net_data_custom_list[i].end(); ++it)
					for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
						buffer_configuration.add_constant_buffer((*it2)->get_size());
				for(auto it = persistent_working_data_list[i].begin(); it != persistent_working_data_list[i].end(); ++it)
					for(auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
						buffer_configuration.add_constant_buffer((*it2)->get_size());

				for(auto it = layer_buffer_set_per_entry_sizes_list[i].begin(); it != layer_buffer_set_per_entry_sizes_list[i].end(); ++it)
					buffer_configuration.add_per_entry_buffer(*it);
				for(auto it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
				{
					// 2 buffers for concurrent input and output data transfer
					buffer_configuration.add_per_entry_buffer(it->second);
					buffer_configuration.add_per_entry_buffer(it->second);
				}
				for(auto it = temporary_working_fixed_set_sizes_list[i].begin(); it != temporary_working_fixed_set_sizes_list[i].end(); ++it)
					buffer_configuration.add_constant_buffer(*it);

				for(auto it = testers_list[i].begin(); it != testers_list[i].end(); ++it)
				{
					std::vector<unsigned int> tex_per_entry = it->second->get_linear_addressing_through_texture_per_entry();
					unsigned int cumulative_tiling_factor = cumulative_tiling_factor_map[it->first];
					for(std::vector<unsigned int>::const_iterator it2 = tex_per_entry.begin(); it2 != tex_per_entry.end(); ++it2)
						buffer_configuration.add_per_entry_linear_addressing_through_texture(*it2 * cumulative_tiling_factor);
				}

				max_entry_count_list[i] = cuda_config->get_max_entry_count(buffer_configuration);

				if (max_entry_count_list[i] == 0)
					throw neural_network_exception((boost::format("Insufficient memory to do forward prop for even one sample on device #%1% %2%") % cuda_config->device_id % cuda_config->device_name).str());

				if (debug->is_debug())
				{
					std::stringstream debug_str;
					debug_str << "forward prop cuda max packet size, device #" << cuda_config->device_id << " " << cuda_config->device_name << ": " << max_entry_count_list[i];
					if (max_entry_count_list[i] > max_max_entry_count)
						debug_str << ", will be capped by " << max_max_entry_count;
					debug->output_message(debug_str.str().c_str());
				}

				max_entry_count_list[i] = std::min(max_entry_count_list[i], max_max_entry_count);
			}

			float fastest_time;
			int fastest_device;
			for(int i = 0; i < max_entry_count_list.size(); ++i)
			{
				float current_time = static_cast<float>(max_entry_count_list[i]) / cuda_multi_config->cuda_config_list[i]->get_flops();
				if ((i == 0) || (current_time < fastest_time))
				{
					fastest_time = current_time;
					fastest_device = i;
				}
			}
			for(int i = 0; i < max_entry_count_list.size(); ++i)
			{
				auto cuda_config = cuda_multi_config->cuda_config_list[i];
				unsigned int new_max_entry_count = static_cast<unsigned int>(cuda_config->get_flops() * fastest_time + 0.5F);
				if (new_max_entry_count != max_entry_count_list[i])
				{
					max_entry_count_list[i] = new_max_entry_count;
					if (debug->is_debug())
					{
						std::stringstream debug_str;
						debug_str << "forward prop cuda changing max packet size to balance load, device #" << cuda_config->device_id << " " << cuda_config->device_name << ": " << max_entry_count_list[i];
						debug->output_message(debug_str.str().c_str());
					}
				}
			}
		}

		float forward_propagation_cuda::get_max_flops() const
		{
			float res = 0.0F;
			for(auto cuda_config: cuda_multi_config->cuda_config_list)
				res += cuda_config->get_flops();

			return res;
		}
	}
}
