/*
 *  Copyright 2011-2015 Maxim Milakov
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

namespace nnforge
{
	namespace cuda
	{
		const unsigned int forward_propagation_cuda::max_max_entry_count = 1024;

		forward_propagation_cuda::forward_propagation_cuda(
			const network_schema& schema,
			const std::vector<std::string>& output_layer_names,
			debug_state::ptr debug,
			cuda_running_configuration::const_ptr cuda_config)
			: forward_propagation(schema, output_layer_names, debug)
			, cuda_config(cuda_config)
			, max_entry_count(0)
		{
			cuda_config->set_device();

			layers_in_forward_propagation_order = this->schema->get_layers_in_forward_propagation_order();

			std::vector<layer::const_ptr> layer_list = this->schema->get_layers();
			for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
				if ((*it)->get_type_name() != data_layer::layer_type_name)
					testing_schemas.insert(
						std::make_pair(
							(*it)->instance_name,
							layer_testing_schema_factory::singleton::get_const_instance().create_testing_schema_layer(*it, cuda_config)));

			setup_network_cuda();

			setup_streams_and_events();

			for(std::map<std::string, layer_testing_schema::const_ptr>::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it)
				schema_data.insert(std::make_pair(it->first, it->second->get_schema_buffers()));
		}

		forward_propagation_cuda::~forward_propagation_cuda()
		{
		}

		forward_propagation_cuda::read_entry_info::read_entry_info()
		{
		}

		unsigned int forward_propagation_cuda::actual_run(
			structured_data_bunch_reader& reader,
			structured_data_bunch_writer& writer)
		{
			cuda_config->set_device();

			unsigned int current_max_entry_count = max_entry_count;
			int approximate_reader_entry_count = reader.get_approximate_entry_count();
			if (approximate_reader_entry_count > 0)
				current_max_entry_count = std::min(current_max_entry_count, static_cast<unsigned int>(approximate_reader_entry_count));
			current_max_entry_count = std::min(current_max_entry_count, max_max_entry_count);

			std::map<std::string, nnforge_array<cuda_linear_buffer_device::ptr, 2> > dedicated_buffers;
			for(std::map<std::string, size_t>::const_iterator it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
			{
				nnforge_array<cuda_linear_buffer_device::ptr, 2>& arr = dedicated_buffers.insert(std::make_pair(it->first, nnforge_array<cuda_linear_buffer_device::ptr, 2>())).first->second;
				arr[0] = cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(it->second * current_max_entry_count));
				arr[1] = cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(it->second * current_max_entry_count));
			}

			std::map<std::string, cuda_linear_buffer_host::ptr> input_host_buffers;
			for(std::map<std::string, size_t>::const_iterator it = input_per_entry_host_data_name_to_size_map.begin(); it != input_per_entry_host_data_name_to_size_map.end(); ++it)
				input_host_buffers.insert(std::make_pair(it->first,
					cuda_linear_buffer_host::ptr(new cuda_linear_buffer_host(it->second * current_max_entry_count))));
			std::map<std::string, cuda_linear_buffer_host::ptr> output_host_buffers;
			for(std::map<std::string, size_t>::const_iterator it = output_per_entry_host_data_name_to_size_map.begin(); it != output_per_entry_host_data_name_to_size_map.end(); ++it)
				output_host_buffers.insert(std::make_pair(it->first,
					cuda_linear_buffer_host::ptr(new cuda_linear_buffer_host(it->second * current_max_entry_count))));
	
			run_kernels_task_ready = false;

			unsigned int entry_processed_count = 0;

			run_kernels_params params(
				dedicated_buffers,
				current_max_entry_count);
			boost::thread run_kernels_thread(run_kernels_static, this, &params);
			try
			{
				run_kernels_thread_io_set = 0;
				bool initial_iteration = true;
				bool try_to_read = true;
				bool run_kernels_thread_stopped = false;
				bool entry_not_read_encountered = false;
				unsigned int entry_to_process_count = 0;
				unsigned int entry_to_write_count = 0;
				unsigned int base_entry_to_read_id = 0;
				std::vector<read_entry_info> read_entry_info_list(current_max_entry_count);
				for(unsigned int i = 0; i < current_max_entry_count; ++i)
				{
					for(std::map<std::string, size_t>::const_iterator it = input_per_entry_host_data_name_to_size_map.begin(); it != input_per_entry_host_data_name_to_size_map.end(); ++it)
					{
						read_entry_info_list[i].data_map.insert(std::make_pair(it->first, (float *)(*input_host_buffers[it->first]) + i * (it->second / sizeof(float))));
						read_entry_info_list[i].reader = &reader;
					}
				}

				while(true)
				{
					unsigned int copy_data_thread_io_set = 1 - run_kernels_thread_io_set;
					bool wait_for_kernels_to_finish = false;
					if (!initial_iteration && !run_kernels_thread_stopped)
					{
						// Set command
						run_kernels_thread_entry_to_process_count = entry_to_process_count;
						run_kernels_finished = false;
						{
							boost::lock_guard<boost::mutex> lock(run_kernels_pending_mutex);
							run_kernels_task_ready = true;
						}
						run_kernels_pending_condition.notify_one();
						run_kernels_thread_stopped = (run_kernels_thread_entry_to_process_count == 0);
						wait_for_kernels_to_finish = !run_kernels_thread_stopped;
					}

					// Launch D2H copy for output data
					if (entry_to_write_count > 0)
					{
						for(std::map<std::string, cuda_linear_buffer_host::ptr>::iterator it = output_host_buffers.begin(); it != output_host_buffers.end(); ++it)
						{
							cuda_safe_call(cudaMemcpyAsync(
								*it->second,
								*dedicated_buffers[it->first][copy_data_thread_io_set],
								output_per_entry_host_data_name_to_size_map[it->first] * entry_to_write_count,
								cudaMemcpyDeviceToHost,
								*copy_data_stream));
						}
						if (cuda_config->is_flush_required())
							cuda_safe_call(cudaStreamQuery(*copy_data_stream));
					}

					unsigned int entry_read_count = 0;
					if (!entry_not_read_encountered)
					{
						PUSH_RANGE("Reading input data", 0);
						// Launch all read input data tasks
						for(unsigned int i = 0; i < current_max_entry_count; ++i)
						{
							read_entry_info& current_info = read_entry_info_list[i];
							current_info.read_entry_finished = false;
							current_info.entry_id = base_entry_to_read_id + i;
							cuda_config->get_job_runner()->service.post(boost::bind(read_input_data_static, &current_info));
						}

						// Wait for all input data to be read
						for(unsigned int i = 0; i < current_max_entry_count; ++i)
						{
							read_entry_info& current_info = read_entry_info_list[i];

							{
								boost::unique_lock<boost::mutex> lock(current_info.read_entry_finished_mutex);
								while (!current_info.read_entry_finished)
									current_info.read_entry_finished_condition.wait(lock);
							}
							if (!current_info.error_message.empty())
							{
								for(unsigned int j = i; j < current_max_entry_count; ++j)
								{
									read_entry_info& current_info = read_entry_info_list[j];
									{
										boost::unique_lock<boost::mutex> lock(current_info.read_entry_finished_mutex);
										while (!current_info.read_entry_finished)
											current_info.read_entry_finished_condition.wait(lock);
									}
								}
								throw neural_network_exception(params.error_message);
							}
							if (!entry_not_read_encountered)
							{
								if (current_info.entry_read)
									++entry_read_count;
								else
									entry_not_read_encountered = true;
							}
						}
						POP_RANGE;
					} // if (!entry_not_read_encountered)

					// Make sure output data is copied to host
					cuda_safe_call(cudaStreamSynchronize(*copy_data_stream));

					// Launch H2D copy for input data
					if (entry_read_count > 0)
					{
						for(std::map<std::string, cuda_linear_buffer_host::ptr>::iterator it = input_host_buffers.begin(); it != input_host_buffers.end(); ++it)
						{
							cuda_safe_call(cudaMemcpyAsync(
								*dedicated_buffers[it->first][copy_data_thread_io_set],
								*it->second,
								input_per_entry_host_data_name_to_size_map[it->first] * entry_read_count,
								cudaMemcpyDeviceToHost,
								*copy_data_stream));
						}
						if (cuda_config->is_flush_required())
							cuda_safe_call(cudaStreamQuery(*copy_data_stream));
					}

					// Write output data
					if (entry_to_write_count > 0)
					{
						PUSH_RANGE("Writing output data", 1);
						for(unsigned int i = 0; i < entry_to_write_count; ++i)
						{
							std::map<std::string, const float *> data_map;
							for(std::map<std::string, size_t>::const_iterator it = output_per_entry_host_data_name_to_size_map.begin(); it != output_per_entry_host_data_name_to_size_map.end(); ++it)
								data_map.insert(std::make_pair(it->first, (float *)(*output_host_buffers[it->first]) + i * (it->second / sizeof(float))));
							writer.write(data_map);
						}
						POP_RANGE;
					}

					// Make sure input data is copied to device
					cuda_safe_call(cudaStreamSynchronize(*copy_data_stream));

					if (wait_for_kernels_to_finish)
					{
						PUSH_RANGE("Waiting for kernels to finish", 2);
						// Wait for all the kernels to finish execution
						{
							boost::unique_lock<boost::mutex> lock(run_kernels_finished_mutex);
							while (!run_kernels_finished)
								run_kernels_finished_condition.wait(lock);
						}
						POP_RANGE;
						if (!params.error_message.empty())
							throw neural_network_exception(params.error_message);
					}

					run_kernels_thread_io_set = 1 - run_kernels_thread_io_set; // Switch set of IO buffers
					initial_iteration = false;
					entry_processed_count += entry_to_write_count;
					base_entry_to_read_id += entry_read_count;
					entry_to_write_count = entry_to_process_count;
					entry_to_process_count = entry_read_count;

					if ((entry_read_count == 0) && (!wait_for_kernels_to_finish))
						break;
				}
			}
			catch (const std::exception&)
			{
				run_kernels_thread.interrupt();
				run_kernels_thread.join();
				throw;
			}

			run_kernels_thread.join();
			if (!params.error_message.empty())
				throw neural_network_exception(params.error_message);

			return entry_processed_count;
		}

		void forward_propagation_cuda::read_input_data_static(read_entry_info * params)
		{
			try
			{
				params->entry_read = params->reader->read(params->entry_id, params->data_map);

				// Notify caller thread that result is ready
				{
					boost::lock_guard<boost::mutex> lock(params->read_entry_finished_mutex);
					params->read_entry_finished = true;
				}
				params->read_entry_finished_condition.notify_one();
			}
			catch (const std::runtime_error& e)
			{
				params->error_message = e.what();
				{
					boost::lock_guard<boost::mutex> lock(params->read_entry_finished_mutex);
					params->read_entry_finished = true;
				}
				params->read_entry_finished_condition.notify_one();
			}
		}

		void forward_propagation_cuda::run_kernels(run_kernels_params& params)
		{
			try
			{
				cuda_config->set_device();

				std::vector<cuda_linear_buffer_device::ptr> temporary_working_fixed_buffers;
				for(std::vector<size_t>::const_iterator it = temporary_working_fixed_set_size_list.begin(); it != temporary_working_fixed_set_size_list.end(); ++it)
					temporary_working_fixed_buffers.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(*it)));

				std::vector<cuda_linear_buffer_device::ptr> temporary_working_per_entry_buffers;
				for(std::vector<size_t>::const_iterator it = temporary_working_per_entry_set_size_list.begin(); it != temporary_working_per_entry_set_size_list.end(); ++it)
					temporary_working_per_entry_buffers.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(*it * params.current_max_entry_count)));

				std::vector<cuda_linear_buffer_device::ptr> layer_buffers;
				for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
					layer_buffers.push_back(cuda_linear_buffer_device::ptr(new cuda_linear_buffer_device(*it * params.current_max_entry_count)));

				boost::unique_lock<boost::mutex> lock(run_kernels_pending_mutex);
				while(true)
				{
					boost::this_thread::interruption_point();

					while (!run_kernels_task_ready)
						run_kernels_pending_condition.wait(lock);

					run_kernels_task_ready = false;

					if (run_kernels_thread_entry_to_process_count == 0)
						break;

					for(std::vector<layer::const_ptr>::const_iterator layer_it = layers_in_forward_propagation_order.begin(); layer_it != layers_in_forward_propagation_order.end(); ++layer_it)
					{
						layer::const_ptr current_layer = *layer_it;
						const std::string& layer_name = current_layer->instance_name;

						if (data_layer_names.find(layer_name) != data_layer_names.end())
							continue;

						cuda_stream::ptr current_stream = command_streams[layer_to_stream_set_map[layer_name]];

						// Enqueue waits for previous events
						{
							std::map<std::string, std::vector<cuda_event::ptr> >::const_iterator previous_events_it = layer_previous_events.find(layer_name);
							if (previous_events_it != layer_previous_events.end())
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
								std::map<std::string, unsigned int>::const_iterator it = layer_buffer_name_to_set_map.find(layer_name);
								if (it != layer_buffer_name_to_set_map.end())
									output_buffer = layer_buffers[it->second];
								else
									output_buffer = params.dedicated_buffers.find(layer_name)->second[run_kernels_thread_io_set];
							}

							std::vector<cuda_linear_buffer_device::const_ptr> input_buffers;
							for(std::vector<std::string>::const_iterator input_layer_name_it = current_layer->input_layer_instance_names.begin(); input_layer_name_it != current_layer->input_layer_instance_names.end(); ++input_layer_name_it)
							{
								std::map<std::string, unsigned int>::const_iterator it = layer_buffer_name_to_set_map.find(*input_layer_name_it);
								if (it != layer_buffer_name_to_set_map.end())
									input_buffers.push_back(layer_buffers[it->second]);
								else
									input_buffers.push_back(params.dedicated_buffers.find(*input_layer_name_it)->second[run_kernels_thread_io_set]);
							}

							cuda_linear_buffer_device::ptr temporary_working_fixed_buffer;
							{
								std::map<std::string, unsigned int>::const_iterator it = temporary_working_fixed_data_name_to_set_map.find(layer_name);
								if (it != temporary_working_fixed_data_name_to_set_map.end())
									temporary_working_fixed_buffer = temporary_working_fixed_buffers[it->second];
							}

							cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer;
							{
								std::map<std::string, unsigned int>::const_iterator it = temporary_working_per_entry_data_name_to_set_map.find(layer_name);
								if (it != temporary_working_per_entry_data_name_to_set_map.end())
									temporary_working_per_entry_buffer = temporary_working_per_entry_buffers[it->second];
							}

							std::vector<cuda_linear_buffer_device::const_ptr> data_list;
							{
								std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >::const_iterator data_list_it = net_data.find(layer_name);
								if (data_list_it != net_data.end())
									data_list = data_list_it->second;
							}

							std::vector<cuda_linear_buffer_device::const_ptr> data_custom_list;
							{
								std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >::const_iterator data_custom_list_it = net_data_custom.find(layer_name);
								if (data_custom_list_it != net_data_custom.end())
									data_custom_list = data_custom_list_it->second;
							}

							testers.find(layer_name)->second->enqueue_forward_propagation(
								*current_stream,
								output_buffer,
								schema_data[layer_name],
								data_list,
								data_custom_list,
								input_buffers,
								persistent_working_data[layer_name],
								temporary_working_fixed_buffer,
								temporary_working_per_entry_buffer,
								run_kernels_thread_entry_to_process_count * cumulative_tiling_factor_map[layer_name]);
						}

						// Enqeue event
						{
							std::map<std::string, cuda_event::ptr>::const_iterator current_event_it = layer_output_data_ready_events.find(layer_name);
							if (current_event_it != layer_output_data_ready_events.end())
								cudaEventRecord(*current_event_it->second, *current_stream);
						}

						if (cuda_config->is_flush_required())
							cuda_safe_call(cudaStreamQuery(*current_stream));
					}

					// Wait for output data to be ready
					for(std::vector<cuda_event::ptr>::const_iterator event_it = output_data_ready_additional_events.begin(); event_it != output_data_ready_additional_events.end(); ++event_it)
						cuda_safe_call(cudaStreamWaitEvent(*command_streams[output_data_ready_stream_set_id], **event_it, 0));
					cudaStreamSynchronize(*command_streams[output_data_ready_stream_set_id]);
		
					// Notify caller thread that result is ready
					{
						boost::lock_guard<boost::mutex> lock(run_kernels_finished_mutex);
						run_kernels_finished = true;
					}
					run_kernels_finished_condition.notify_one();
				}
			}
			catch (const std::runtime_error& e)
			{
				params.error_message = e.what();
				{
					boost::lock_guard<boost::mutex> lock(run_kernels_finished_mutex);
					run_kernels_finished = true;
				}
				run_kernels_finished_condition.notify_one();
			}
		}

		void forward_propagation_cuda::run_kernels_static(forward_propagation_cuda * self, run_kernels_params * params)
		{
			self->run_kernels(*params);
		}

		forward_propagation_cuda::run_kernels_params::run_kernels_params(
			std::map<std::string, nnforge_array<cuda_linear_buffer_device::ptr, 2> >& dedicated_buffers,
			unsigned int current_max_entry_count)
			: dedicated_buffers(dedicated_buffers)
			, current_max_entry_count(current_max_entry_count)
		{
		}

		void forward_propagation_cuda::setup_network_cuda()
		{
			copy_data_stream = cuda_stream::ptr(new cuda_stream());
		}

		void forward_propagation_cuda::setup_streams_and_events()
		{
			command_streams.clear();
			layer_to_stream_set_map.clear();
			layer_output_data_ready_events.clear();
			layer_previous_events.clear();
			output_data_ready_additional_events.clear();

			std::vector<std::vector<layer::const_ptr> > layer_stream_set = schema->get_layer_stream_set_for_forward_propagation();
			command_streams.resize(layer_stream_set.size());
			for(unsigned int stream_set_id = 0; stream_set_id < static_cast<unsigned int>(layer_stream_set.size()); ++stream_set_id)
			{
				command_streams[stream_set_id] = cuda_stream::ptr(new cuda_stream());
				for(std::vector<layer::const_ptr>::const_iterator it = layer_stream_set[stream_set_id].begin(); it != layer_stream_set[stream_set_id].end(); ++it)
					layer_to_stream_set_map.insert(std::make_pair((*it)->instance_name, stream_set_id));
			}
			if (debug->is_debug())
			{
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_cuda_streams", "dot"), std::ios_base::out | std::ios_base::trunc);
				schema->write_dot(out, layer_to_stream_set_map);
			}

			for(std::vector<layer::const_ptr>::const_reverse_iterator it = layers_in_forward_propagation_order.rbegin(); it != layers_in_forward_propagation_order.rend(); ++it)
			{
				layer::const_ptr current_layer = *it;

				if (data_layer_names.find(current_layer->instance_name) != data_layer_names.end())
					continue;

				unsigned int current_stream_set_id = layer_to_stream_set_map.find(current_layer->instance_name)->second;

				std::vector<cuda_event::ptr> previous_events;
				for(std::vector<std::string>::const_iterator it2 = current_layer->input_layer_instance_names.begin(); it2 != current_layer->input_layer_instance_names.end(); ++it2)
				{
					const std::string& previous_layer_name = *it2;
					if (data_layer_names.find(previous_layer_name) != data_layer_names.end()) 
						continue;
					unsigned int previous_stream_set_id = layer_to_stream_set_map.find(previous_layer_name)->second;
					if (previous_stream_set_id == current_stream_set_id)
						continue;

					cuda_event::ptr previous_event;
					std::map<std::string, cuda_event::ptr>::const_iterator it3 = layer_output_data_ready_events.find(previous_layer_name);
					if (it3 != layer_output_data_ready_events.end())
						previous_event = it3->second;
					else
						previous_event = layer_output_data_ready_events.insert(std::make_pair(previous_layer_name, cuda_event::ptr(new cuda_event()))).first->second;
					previous_events.push_back(previous_event);
				}

				if (!previous_events.empty())
					layer_previous_events.insert(std::make_pair(current_layer->instance_name, previous_events));
			}

			bool output_data_ready_stream_set_id_defined = false;
			for(std::vector<std::string>::const_iterator it = output_layer_names.begin(); it != output_layer_names.end(); ++it)
			{
				if (!output_data_ready_stream_set_id_defined)
				{
					output_data_ready_stream_set_id = layer_to_stream_set_map[*it];
					output_data_ready_stream_set_id_defined = true;
					continue;
				}
				else
				{
					if (layer_to_stream_set_map.find(*it)->second == output_data_ready_stream_set_id)
						continue;
				}

				cuda_event::ptr previous_event;
				std::map<std::string, cuda_event::ptr>::const_iterator it3 = layer_output_data_ready_events.find(*it);
				if (it3 != layer_output_data_ready_events.end())
					previous_event = it3->second;
				else
					previous_event = layer_output_data_ready_events.insert(std::make_pair(*it, cuda_event::ptr(new cuda_event()))).first->second;
				output_data_ready_additional_events.push_back(previous_event);
			}
		}

		void forward_propagation_cuda::actual_set_data(network_data::const_ptr data)
		{
			cuda_config->set_device();

			host_net_data = data;

			update_data();
		}

		void forward_propagation_cuda::actual_clear_data()
		{
			cuda_config->set_device();

			host_net_data.reset();

			update_data();
		}

		void forward_propagation_cuda::layer_config_map_modified()
		{
			cuda_config->set_device();

			testers.clear();

			setup_io_host_buffer_sizes();

			setup_dedicated_buffer_sizes();

			std::vector<layer::const_ptr> layer_list = schema->get_layers();
			for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
			{
				layer::const_ptr l = *it;

				if (data_layer_names.find(l->instance_name) != data_layer_names.end())
					continue;

				std::vector<layer_configuration_specific> input_layer_configuration_specific_list;
				for(std::vector<std::string>::const_iterator it2 = l->input_layer_instance_names.begin(); it2 != l->input_layer_instance_names.end(); ++it2)
					input_layer_configuration_specific_list.push_back(layer_config_map.find(*it2)->second);

				layer_configuration_specific output_layer_configuration_specific = layer_config_map.find(l->instance_name)->second;

				layer_testing_schema::const_ptr sch = testing_schemas.find(l->instance_name)->second;

				testers.insert(
					std::make_pair(
						l->instance_name,
						sch->create_tester(
							input_layer_configuration_specific_list,
							output_layer_configuration_specific)));
			}

			setup_layer_buffer_sizes();

			setup_temporary_working_fixed_buffer_sizes();

			setup_temporary_working_per_entry_buffer_sizes();

			update_data();

			update_max_entry_count();
		}

		void forward_propagation_cuda::setup_layer_buffer_sizes()
		{
			std::vector<std::vector<layer::const_ptr> > layer_buffer_set_list;
			{
				std::map<std::string, unsigned int> input_index_layer_can_write_output_map;
				for(std::map<std::string, layer_tester_cuda::ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
				{
					int input_index_layer_can_write = it->second->get_input_index_layer_can_write();
					if (input_index_layer_can_write >= 0)
						input_index_layer_can_write_output_map.insert(std::make_pair(it->first, static_cast<unsigned int>(input_index_layer_can_write)));
				}
				std::set<std::string> separate_buffers_layer_names(output_layer_names.begin(), output_layer_names.end());
				separate_buffers_layer_names.insert(data_layer_names.begin(), data_layer_names.end());
				layer_buffer_set_list = schema->get_layer_buffer_set_for_forward_propagation(input_index_layer_can_write_output_map, separate_buffers_layer_names);
			}

			layer_buffer_set_per_entry_size_list.clear();
			layer_buffer_name_to_set_map.clear();
			for(unsigned int set_id = 0; set_id < layer_buffer_set_list.size(); ++set_id)
			{
				const std::vector<layer::const_ptr>& layer_list = layer_buffer_set_list[set_id];
				size_t max_buffer_size_per_entry = 0;
				for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
				{
					const std::string& layer_name = (*it)->instance_name;
					layer_buffer_name_to_set_map.insert(std::make_pair(layer_name, set_id));
					size_t buffer_size_per_entry = layer_config_map.find(layer_name)->second.get_neuron_count() * cumulative_tiling_factor_map[layer_name] * sizeof(float);
					max_buffer_size_per_entry = std::max(max_buffer_size_per_entry, buffer_size_per_entry);
				}
				layer_buffer_set_per_entry_size_list.push_back(max_buffer_size_per_entry);
			}
			if (debug->is_debug())
			{
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_cuda_layer_buffers", "dot"), std::ios_base::out | std::ios_base::trunc);
				schema->write_dot(out, layer_buffer_name_to_set_map);
			}
		}

		void forward_propagation_cuda::setup_temporary_working_fixed_buffer_sizes()
		{
			std::vector<std::vector<layer::const_ptr> > temporary_working_fixed_buffer_set_list;
			{
				std::set<std::string> temporary_working_fixed_layer_set;
				for(std::map<std::string, layer_tester_cuda::ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
				{
					size_t temporary_working_fixed_buffer_size = it->second->get_temporary_working_fixed_buffer_size();
					if (temporary_working_fixed_buffer_size > 0)
						temporary_working_fixed_layer_set.insert(it->first);
				}
				temporary_working_fixed_buffer_set_list = schema->get_temporary_working_buffer_set_for_forward_propagation(temporary_working_fixed_layer_set);
			}

			temporary_working_fixed_set_size_list.clear();
			temporary_working_fixed_data_name_to_set_map.clear();
			for(unsigned int set_id = 0; set_id < temporary_working_fixed_buffer_set_list.size(); ++set_id)
			{
				const std::vector<layer::const_ptr>& layer_list = temporary_working_fixed_buffer_set_list[set_id];
				size_t max_buffer_size = 0;
				for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
				{
					const std::string& layer_name = (*it)->instance_name;
					temporary_working_fixed_data_name_to_set_map.insert(std::make_pair(layer_name, set_id));
					size_t buffer_size = testers.find(layer_name)->second->get_temporary_working_fixed_buffer_size();
					max_buffer_size = std::max(max_buffer_size, buffer_size);
				}
				temporary_working_fixed_set_size_list.push_back(max_buffer_size);
			}
			if (debug->is_debug())
			{
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_cuda_temporary_fixed_buffers", "dot"), std::ios_base::out | std::ios_base::trunc);
				schema->write_dot(out, temporary_working_fixed_data_name_to_set_map);
			}
		}

		void forward_propagation_cuda::setup_temporary_working_per_entry_buffer_sizes()
		{
			std::vector<std::vector<layer::const_ptr> > temporary_working_per_entry_buffer_set_list;
			{
				std::set<std::string> temporary_working_per_entry_layer_set;
				for(std::map<std::string, layer_tester_cuda::ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
				{
					size_t temporary_working_per_entry_buffer_size = it->second->get_temporary_working_per_entry_buffer_size();
					if (temporary_working_per_entry_buffer_size > 0)
						temporary_working_per_entry_layer_set.insert(it->first);
				}
				temporary_working_per_entry_buffer_set_list = schema->get_temporary_working_buffer_set_for_forward_propagation(temporary_working_per_entry_layer_set);
			}

			temporary_working_per_entry_set_size_list.clear();
			temporary_working_per_entry_data_name_to_set_map.clear();
			for(unsigned int set_id = 0; set_id < temporary_working_per_entry_buffer_set_list.size(); ++set_id)
			{
				const std::vector<layer::const_ptr>& layer_list = temporary_working_per_entry_buffer_set_list[set_id];
				size_t max_buffer_size = 0;
				for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
				{
					const std::string& layer_name = (*it)->instance_name;
					temporary_working_per_entry_data_name_to_set_map.insert(std::make_pair(layer_name, set_id));
					size_t buffer_size = testers.find(layer_name)->second->get_temporary_working_per_entry_buffer_size() * cumulative_tiling_factor_map[layer_name];
					max_buffer_size = std::max(max_buffer_size, buffer_size);
				}
				temporary_working_per_entry_set_size_list.push_back(max_buffer_size);
			}
			if (debug->is_debug())
			{
				boost::filesystem::ofstream out(debug->get_path_to_unique_file("forward_prop_cuda_temporary_per_entry_buffers", "dot"), std::ios_base::out | std::ios_base::trunc);
				schema->write_dot(out, temporary_working_per_entry_data_name_to_set_map);
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
			cuda_config->set_device();

			net_data.clear();
			net_data_custom.clear();
			persistent_working_data.clear();

			if (testers.empty() || (!host_net_data))
				return;

			for(std::map<std::string, layer_tester_cuda::ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
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

		void forward_propagation_cuda::update_max_entry_count()
		{
			buffer_cuda_size_configuration buffer_configuration;

			for(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >::const_iterator it = schema_data.begin(); it != schema_data.end(); ++it)
				for(std::vector<cuda_linear_buffer_device::const_ptr>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());
			for(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
				for(std::vector<cuda_linear_buffer_device::const_ptr>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());
			for(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >::const_iterator it = net_data_custom.begin(); it != net_data_custom.end(); ++it)
				for(std::vector<cuda_linear_buffer_device::const_ptr>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());
			for(std::map<std::string, std::vector<cuda_linear_buffer_device::const_ptr> >::const_iterator it = persistent_working_data.begin(); it != persistent_working_data.end(); ++it)
				for(std::vector<cuda_linear_buffer_device::const_ptr>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<size_t>::const_iterator it = layer_buffer_set_per_entry_size_list.begin(); it != layer_buffer_set_per_entry_size_list.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it);
			for(std::map<std::string, size_t>::const_iterator it = dedicated_per_entry_data_name_to_size_map.begin(); it != dedicated_per_entry_data_name_to_size_map.end(); ++it)
			{
				// 2 buffers for concurrent input and output data transfer
				buffer_configuration.add_per_entry_buffer(it->second);
				buffer_configuration.add_per_entry_buffer(it->second);
			}
			for(std::vector<size_t>::const_iterator it = temporary_working_fixed_set_size_list.begin(); it != temporary_working_fixed_set_size_list.end(); ++it)
				buffer_configuration.add_constant_buffer(*it);
			for(std::vector<size_t>::const_iterator it = temporary_working_per_entry_set_size_list.begin(); it != temporary_working_per_entry_set_size_list.end(); ++it)
				buffer_configuration.add_per_entry_buffer(*it);

			for(std::map<std::string, layer_tester_cuda::ptr>::const_iterator it = testers.begin(); it != testers.end(); ++it)
			{
				std::vector<unsigned int> tex_per_entry = it->second->get_linear_addressing_through_texture_per_entry();
				unsigned int cumulative_tiling_factor = cumulative_tiling_factor_map[it->first];
				for(std::vector<unsigned int>::const_iterator it2 = tex_per_entry.begin(); it2 != tex_per_entry.end(); ++it2)
					buffer_configuration.add_per_entry_linear_addressing_through_texture(*it2 * cumulative_tiling_factor);
			}

			max_entry_count = cuda_config->get_max_entry_count(buffer_configuration);

			if (debug->is_debug())
				debug->output_message((boost::format("forward prop cuda max packet size %1%, will be capped by %2%") % max_entry_count % max_max_entry_count).str().c_str());
		}
	}
}
