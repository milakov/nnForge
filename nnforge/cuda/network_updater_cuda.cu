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

#include "network_updater_cuda.h"

#include "neural_network_cuda_exception.h"
#include "layer_testing_schema_factory.h"
#include "cuda_linear_buffer_device.h"
#include "cuda_linear_buffer_host.h"
#include "util_cuda.h"
#include "cuda_event.h"
#include "layer_updater_schema_factory.h"
#include "supervised_data_reader_async_helper.h"
#include "error_function_updater_cuda_factory.h"

#include "../nn_types.h"

#include <cuda_runtime.h>
#include <boost/format.hpp>
#include <stack>
#include <numeric>

#include "../debug_util.h"
#include <boost/filesystem.hpp>

namespace nnforge
{
	namespace cuda
	{
		__forceinline__ __device__ double atomicAdd(double* address, double val)
		{
				unsigned long long int* address_as_ull = (unsigned long long int*)address;
				unsigned long long int old = *address_as_ull, assumed;
				do {
					assumed = old;
					old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
				} while (assumed != old);
				return __longlong_as_double(old);
		}

		__global__ void convert_compacted_to_raw_upd_kernel(
			const uchar4 * __restrict input,
			float4 * __restrict output,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				uchar4 inp = input[elem_id];
				float4 val;
				val.x = inp.x * (1.0F / 255.0F);
				val.y = inp.y * (1.0F / 255.0F);
				val.z = inp.z * (1.0F / 255.0F);
				val.w = inp.w * (1.0F / 255.0F);
				output[elem_id] = val;
			}
		}

		__global__ void dropout_kernel(
			float * __restrict neurons,
			const float * __restrict random_buf,
			float dropout_rate,
			float scale,
			int offset,
			unsigned int mask,
			int elem_count)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			if (elem_id < elem_count)
			{
				float val = neurons[elem_id];
				unsigned int random_elem_id = (elem_id + offset) & mask;
				bool under_thresgold = (random_buf[random_elem_id] < dropout_rate);
				val *= under_thresgold ? 0.0F : scale;
				neurons[elem_id] = val;
			}
		}

		extern __shared__ float arr_sh[];
		__global__ void apply_gradient_kernel(
			float * __restrict data,
			float * __restrict gradient,
			double * __restrict update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			int elem_count,
			unsigned int update_accum_mask)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			float upd_acc = 0.0F;
			if (elem_id < elem_count)
			{
				float current_weight = __load_nc(data + elem_id);
				float gr = __load_nc(gradient + elem_id);
				float upd = learning_rate * (gr * normalizer - current_weight * weight_decay);
				float new_weight = current_weight + upd;
				data[elem_id] = new_weight;
				gradient[elem_id] = 0.0F;
				upd_acc = fabs(upd);
			}

			int thread_id = threadIdx.x;
			int lane_id = thread_id & 31;
		#if __CUDA_ARCH__ < 300
			volatile float * arr = arr_sh;
			arr[thread_id] = upd_acc;
		#endif
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
			#if __CUDA_ARCH__ < 300
				if (lane_id < tx)
					arr[thread_id] += arr[thread_id + tx];
			#else
				upd_acc += __shfl_down(upd_acc, tx);
			#endif
			}
		#if __CUDA_ARCH__ < 300
			upd_acc = arr[thread_id];
			__syncthreads();
		#endif

			if (blockDim.x > 32)
			{
				if (lane_id == 0)
					arr_sh[thread_id >> 5] = upd_acc;
				__syncthreads();
			}

			if (thread_id == 0)
			{
				for(int i = 1; i < (blockDim.x >> 5); ++i)
					upd_acc += arr_sh[i];
				double upd_acc_d = (double)upd_acc;

				int accum_bucket_id = (elem_id >> 5) & update_accum_mask;

				atomicAdd(update_accum + accum_bucket_id, upd_acc_d);
			}
		}

		__global__ void apply_gradient_with_momentum_kernel(
			float * __restrict data,
			float * __restrict gradient,
			float * __restrict previous_upd,
			double * __restrict update_accum,
			float learning_rate,
			float normalizer,
			float weight_decay,
			float momentum,
			int elem_count,
			unsigned int update_accum_mask)
		{
			int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
			float upd_acc = 0.0F;
			if (elem_id < elem_count)
			{
				float current_weight = __load_nc(data + elem_id);
				float gr = __load_nc(gradient + elem_id);
				float prev_upd = __load_nc(previous_upd + elem_id);
				float upd = prev_upd * momentum + learning_rate * (gr * normalizer - current_weight * weight_decay);
				float new_weight = current_weight + upd;
				data[elem_id] = new_weight;
				gradient[elem_id] = 0.0F;
				previous_upd[elem_id] = upd;
				upd_acc = fabs(upd);
			}

			int thread_id = threadIdx.x;
			int lane_id = thread_id & 31;
		#if __CUDA_ARCH__ < 300
			volatile float * arr = arr_sh;
			arr[thread_id] = upd_acc;
		#endif
			#pragma unroll
			for(int tx = 16; tx > 0; tx >>= 1)
			{
			#if __CUDA_ARCH__ < 300
				if (lane_id < tx)
					arr[thread_id] += arr[thread_id + tx];
			#else
				upd_acc += __shfl_down(upd_acc, tx);
			#endif
			}
		#if __CUDA_ARCH__ < 300
			upd_acc = arr[thread_id];
			__syncthreads();
		#endif

			if (blockDim.x > 32)
			{
				if (lane_id == 0)
					arr_sh[thread_id >> 5] = upd_acc;
				__syncthreads();
			}

			if (thread_id == 0)
			{
				for(int i = 1; i < (blockDim.x >> 5); ++i)
					upd_acc += arr_sh[i];
				double upd_acc_d = (double)upd_acc;

				int accum_bucket_id = (elem_id >> 5) & update_accum_mask;

				atomicAdd(update_accum + accum_bucket_id, upd_acc_d);
			}
		}

		const unsigned int network_updater_cuda::max_entry_count_in_single_batch = 1024;
		const unsigned int network_updater_cuda::elem_count_update_accum_per_part = 64;

		network_updater_cuda::network_updater_cuda(
			network_schema_smart_ptr schema,
			const_error_function_smart_ptr ef,
			cuda_running_configuration_const_smart_ptr cuda_config)
			: network_updater(schema, ef)
			, cuda_config(cuda_config)
		{
			cuda_config->set_device();

			const const_layer_list& layer_list = *schema;

			testing_layer_count = 0;
			start_layer_nonempty_weights_iterator = layer_list.begin();
			for(const_layer_list::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
			{
				start_layer_nonempty_weights_iterator = it;

				if (!(*it)->is_empty_data())
					break;

				testing_layer_count++;
			}

			ef_updater = single_error_function_updater_cuda_factory::get_const_instance().get_error_function_updater_cuda(ef->get_uuid());

			error_function_fused_with_activation = (ef_updater->get_fusable_activation_uuid() == layer_list.back()->get_uuid());

			for(const_layer_list::const_iterator it = layer_list.begin(); it != start_layer_nonempty_weights_iterator; ++it)
				testing_schemas.push_back(single_layer_testing_schema_factory::get_const_instance().create_testing_schema_layer(*it, cuda_config));

			for(const_layer_list::const_iterator it = start_layer_nonempty_weights_iterator; it != layer_list.end(); ++it)
			{
				if ((it != layer_list.end() - 1) || (!error_function_fused_with_activation))
					updater_schemas.push_back(single_layer_updater_schema_factory::get_const_instance().create_updater_schema_layer(*it, cuda_config));
			}

			setup_network_cuda();

			for(const_layer_testing_schema_list::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it)
				testing_schema_data.push_back((*it)->get_schema_buffers());

			for(const_layer_updater_schema_list::const_iterator it = updater_schemas.begin(); it != updater_schemas.end(); ++it)
				updater_schema_data.push_back((*it)->get_schema_buffers());
		}

		network_updater_cuda::~network_updater_cuda()
		{
		}

		void network_updater_cuda::setup_network_cuda()
		{
			command_stream = cuda_stream_smart_ptr(new cuda_stream());
			data_stream = cuda_stream_smart_ptr(new cuda_stream());
		}

		std::pair<testing_result_smart_ptr, training_stat_smart_ptr> network_updater_cuda::actual_update(
			supervised_data_reader& reader,
			const std::vector<std::vector<float> >& learning_rates,
			network_data_smart_ptr data,
			unsigned int batch_size,
			float weight_decay,
			float momentum,
			const std::map<unsigned int, float>& layer_to_dropout_rate_map)
		{
			cuda_config->set_device();

			testing_result_smart_ptr testing_res(new testing_result(ef));

			reader.reset();

			layer_configuration_specific input_configuration = reader.get_input_configuration();
			layer_configuration_specific output_configuration = reader.get_output_configuration();

			unsigned int input_neuron_count = input_configuration.get_neuron_count();
			unsigned int output_neuron_count = output_configuration.get_neuron_count();
			unsigned int input_neuron_count_per_feature_map = input_configuration.get_neuron_count_per_feature_map();
			unsigned int output_neuron_count_per_feature_map = output_configuration.get_neuron_count_per_feature_map();
			neuron_data_type::input_type type_code = reader.get_input_type();
			size_t input_neuron_elem_size = reader.get_input_neuron_elem_size();

			if (error_function_fused_with_activation && (output_neuron_count_per_feature_map != 1))
				throw neural_network_exception("Error function is fused with activation but output_neuron_count_per_feature_map is not equal 1: not implemented");

			unsigned int part_count = 0;
			for(layer_data_list::const_iterator it = data->data_list.begin(); it != data->data_list.end(); ++it)
				part_count += (*it)->size();
			unsigned int elem_count_update_accum = part_count * elem_count_update_accum_per_part;

			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > net_data = get_data(data);
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > net_data_custom = set_get_data_custom(data);
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > gradient = get_zero_gradient(net_data);
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > previous_upd;
			if (momentum > 0.0F)
				previous_upd = get_zero_gradient(net_data);

			unsigned int updater_max_count;
			{
				buffer_cuda_size_configuration buffers_config;

				buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // initial error
				buffers_config.add_constant_buffer(sizeof(double)); // error buffer
				buffers_config.add_constant_buffer(sizeof(double) * elem_count_update_accum); // update_accum
				if (!random_uniform_list.empty())
					buffers_config.add_constant_buffer(random_uniform_list.size() * sizeof(float)); // random_uniform_list
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data_custom.begin(); it != net_data_custom.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = gradient.begin(); it != gradient.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = previous_upd.begin(); it != previous_upd.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());

				for(std::vector<layer_updater_cuda_smart_ptr>::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it)
					(*it)->update_buffer_configuration(buffers_config);

				updater_max_count = cuda_config->get_max_entry_count(buffers_config, 0.9F);
			}

			unsigned int updater_entry_count;
			std::vector<unsigned int> entry_read_count_list;
			unsigned int max_entry_read_count;
			if (updater_max_count > batch_size)
				updater_entry_count = batch_size;
			else
			{
				unsigned int it_count = (batch_size + updater_max_count - 1) / updater_max_count;
				updater_entry_count = (batch_size + it_count - 1) / it_count;
				max_entry_read_count = updater_entry_count;
				unsigned int sum = 0;
				while (sum < batch_size)
				{
					unsigned int new_item = std::min(batch_size - sum, updater_entry_count);
					sum += new_item;
					entry_read_count_list.push_back(new_item);
				}
			}

			{
				buffer_cuda_size_configuration buffers_config;
				update_buffers_configuration(buffers_config, updater_entry_count);

				buffers_config.add_per_entry_linear_addressing_through_texture(layer_config_list[testing_layer_count].get_neuron_count()); // This is for the first updater to safely read input data through the texture
				buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
				buffers_config.add_per_entry_buffer(input_neuron_count * input_neuron_elem_size); // input
				buffers_config.add_per_entry_buffer(input_neuron_count * sizeof(float)); // converted input
				buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
				buffers_config.add_per_entry_buffer(output_neuron_count * sizeof(float)); // output
				buffers_config.add_constant_buffer(output_neuron_count * sizeof(float) * updater_entry_count); // initial error
				buffers_config.add_constant_buffer(sizeof(double)); // error buffer
				buffers_config.add_constant_buffer(sizeof(double) * elem_count_update_accum); // update_accum
				if (!random_uniform_list.empty())
					buffers_config.add_constant_buffer(random_uniform_list.size() * sizeof(float)); // random_uniform_list
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data.begin(); it != net_data.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = net_data_custom.begin(); it != net_data_custom.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = gradient.begin(); it != gradient.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = previous_upd.begin(); it != previous_upd.end(); ++it)
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
						buffers_config.add_constant_buffer((*it2)->get_size());

				unsigned int max_entry_count = std::min(std::min(cuda_config->get_max_entry_count(buffers_config), reader.get_entry_count()), max_entry_count_in_single_batch);
				if (entry_read_count_list.empty() || (max_entry_count >= batch_size))
				{
					unsigned int it_count = std::max(max_entry_count / batch_size, 1U);
					max_entry_read_count = it_count * batch_size;
					entry_read_count_list.clear();
					entry_read_count_list.push_back(max_entry_read_count);
				}
			}

			cuda_linear_buffer_device_smart_ptr input_buf[2] = 
			{
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_read_count * input_neuron_elem_size)),
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(input_neuron_count * max_entry_read_count * input_neuron_elem_size)),
			};

			cuda_linear_buffer_device_smart_ptr output_buf[2] = 
			{
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(output_neuron_count * max_entry_read_count * sizeof(float))),
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(output_neuron_count * max_entry_read_count * sizeof(float))),
			};

			cuda_linear_buffer_device_smart_ptr input_converted_buf(new cuda_linear_buffer_device(input_neuron_count * max_entry_read_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr initial_error_buf(new cuda_linear_buffer_device(output_neuron_count * updater_entry_count * sizeof(float)));

			cuda_linear_buffer_device_smart_ptr error_buf(new cuda_linear_buffer_device(sizeof(double)));

			cuda_linear_buffer_device_smart_ptr update_accum_buf(new cuda_linear_buffer_device(elem_count_update_accum * sizeof(double)));

			cuda_linear_buffer_device_smart_ptr random_uniform_buf;
			if (!random_uniform_list.empty())
			{
				random_uniform_buf = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(random_uniform_list.size() * sizeof(float)));
				cuda_safe_call(cudaMemcpyAsync(*random_uniform_buf, &(*random_uniform_list.begin()), random_uniform_list.size() * sizeof(float), cudaMemcpyHostToDevice, *command_stream));
			}

			cuda_linear_buffer_device_smart_ptr output_buffer = input_converted_buf;
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > > testing_input_and_additional_buffers_pack;
			for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> additional_buffers = (*it)->allocate_additional_buffers(max_entry_read_count);
				testing_input_and_additional_buffers_pack.push_back(std::make_pair(output_buffer, additional_buffers));
				output_buffer = (*it)->get_output_buffer(output_buffer, additional_buffers);
			}
			std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> > updater_input_and_all_buffers_pack;
			for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it)
			{
				layer_updater_cuda::buffer_set all_buffers = (*it)->allocate_all_buffers(updater_entry_count);
				updater_input_and_all_buffers_pack.push_back(std::make_pair(output_buffer, all_buffers));
				output_buffer = all_buffers.output_neurons_buffer;
			}

			std::vector<cuda_linear_buffer_device_smart_ptr> output_errors_buffers;
			cuda_linear_buffer_device_smart_ptr output_errors = initial_error_buf;
			for(std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::reverse_iterator it = updater_input_and_all_buffers_pack.rbegin(); it != updater_input_and_all_buffers_pack.rend(); ++it)
			{
				output_errors_buffers.push_back(output_errors);
				layer_updater_cuda::buffer_set& all_buffers = it->second;

				if (all_buffers.input_errors_buffer != 0)
					output_errors = all_buffers.input_errors_buffer;
			}

			cuda_linear_buffer_host_smart_ptr input_host_buf(new cuda_linear_buffer_host(input_neuron_count * max_entry_read_count * input_neuron_elem_size));
			unsigned char * input = *input_host_buf;
			cuda_linear_buffer_host_smart_ptr output_host_buf(new cuda_linear_buffer_host(output_neuron_count * max_entry_read_count * sizeof(float)));
			float * output = *output_host_buf;

			// set error to zero
			cuda_util::set_with_value(
				*cuda_config,
				(double *)(*error_buf),
				0.0,
				1,
				*command_stream);
			// set update accumulators to zero
			cuda_util::set_with_value(
				*cuda_config,
				(double *)(*update_accum_buf),
				0.0,
				elem_count_update_accum,
				*command_stream);

			unsigned int current_data_slot = 0;
			unsigned int current_command_slot = 1;
			unsigned int entries_available_for_copy_in_count = reader.get_entry_count();
			unsigned int entries_available_for_processing_count = 0;
			cuda_event data_processed_event;
			cuda_event input_copied_event;
			if (cuda_config->is_flush_required())
			{
				cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
				cuda_safe_call(cudaEventQuery(data_processed_event));
			}

			nnforge_uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(random_uniform_list.size() - 1));
			unsigned int mask = static_cast<unsigned int>(random_uniform_list.size() - 1);
			unsigned int entries_processed_count = 0;
			unsigned int entry_read_count_index = 0;
			unsigned int entry_gradient_calculated_count = 0;
			unsigned int gradient_applied_count = 0;
			while((entries_available_for_copy_in_count > 0) || (entries_available_for_processing_count > 0))
			{
				supervised_data_reader_async_helper async_reader;
				if (entries_available_for_copy_in_count > 0)
				{
					unsigned int entries_to_read_count = std::min<unsigned int>(entry_read_count_list[entry_read_count_index], entries_available_for_copy_in_count);
					async_reader.fun = supervised_data_reader_functor(
						entries_to_read_count,
						&reader,
						input,
						output,
						*(input_buf[current_data_slot]),
						*(output_buf[current_data_slot]),
						cuda_config,
						*data_stream);
					async_reader.start();
					entry_read_count_index++;
					if (entry_read_count_index >= entry_read_count_list.size())
						entry_read_count_index = 0;
				}

				if (entries_available_for_processing_count > 0)
				{
					// Convert input
					if (type_code == neuron_data_type::type_byte)
					{
						int elem_count = (input_neuron_count * entries_available_for_processing_count + 3) / 4;
						std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
							*cuda_config,
							elem_count);
						convert_compacted_to_raw_upd_kernel<<<kernel_dims.first, kernel_dims.second, 0, *command_stream>>>(
							*input_buf[current_command_slot],
							*input_converted_buf,
							elem_count);
					}
					else if (type_code == neuron_data_type::type_float)
					{
						cuda_safe_call(cudaMemcpyAsync(
							*input_converted_buf,
							*input_buf[current_command_slot],
							input_neuron_count * entries_available_for_processing_count * sizeof(float),
							cudaMemcpyDeviceToDevice,
							*command_stream));
					}
					else throw neural_network_exception((boost::format("actual_update cannot handle input neurons of type %1%") % type_code).str());

					// Run ann
					{
						std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, std::vector<cuda_linear_buffer_device_smart_ptr> > >::iterator input_and_additional_buffers_pack_it = testing_input_and_additional_buffers_pack.begin();
						std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = testing_schema_data.begin();
						unsigned int layer_id = 0;
						layer_configuration_specific_list::const_iterator layer_config_it = layer_config_list.begin();
						for(std::vector<layer_tester_cuda_smart_ptr>::iterator it = tester_list.begin(); it != tester_list.end(); ++it, ++input_and_additional_buffers_pack_it, ++schema_data_it, ++layer_id, ++layer_config_it)
						{
							std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(layer_id);
							if (dropout_it != layer_to_dropout_rate_map.end())
							{
								unsigned int offset = dist(gen);
								enqueue_dropout(
									*command_stream,
									random_uniform_buf,
									input_and_additional_buffers_pack_it->first,
									dropout_it->second,
									mask,
									entries_available_for_processing_count * layer_config_it->get_neuron_count(),
									offset);
							}

							(*it)->enqueue_test(
								*command_stream,
								*schema_data_it,
								std::vector<const_cuda_linear_buffer_device_smart_ptr>(),
								std::vector<const_cuda_linear_buffer_device_smart_ptr>(),
								input_and_additional_buffers_pack_it->first,
								input_and_additional_buffers_pack_it->second,
								entries_available_for_processing_count);
						}
					}

					// Apply dropout to the input of the first updater layer
					{
						std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(testing_layer_count);
						if (dropout_it != layer_to_dropout_rate_map.end())
						{
							unsigned int offset = dist(gen);
							enqueue_dropout(
								*command_stream,
								random_uniform_buf,
								updater_input_and_all_buffers_pack[0].first,
								dropout_it->second,
								mask,
								entries_available_for_processing_count * layer_config_list[testing_layer_count].get_neuron_count(),
								offset);
						}
					}

					unsigned int base_input_entry_id = 0;
					while(base_input_entry_id < entries_available_for_processing_count)
					{
						std::stack<unsigned int> offset_list;

						unsigned int current_updater_entry_count = std::min(std::min(entries_available_for_processing_count - base_input_entry_id, updater_entry_count), batch_size - entry_gradient_calculated_count);

						// Forward updater
						{
							std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::iterator input_and_all_buffers_pack_it = updater_input_and_all_buffers_pack.begin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator net_data_it = net_data.begin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator net_data_custom_it = net_data_custom.begin();
							std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::iterator schema_data_it = updater_schema_data.begin();
							unsigned int layer_id = testing_layer_count;
							layer_configuration_specific_list::const_iterator layer_config_it = layer_config_list.begin() + testing_layer_count;
							for(std::vector<layer_updater_cuda_smart_ptr>::iterator it = updater_list.begin(); it != updater_list.end(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++net_data_it, ++net_data_custom_it, ++layer_id, ++layer_config_it)
							{
								if (it != updater_list.begin())
								{
									std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(layer_id);
									if (dropout_it != layer_to_dropout_rate_map.end())
									{
										unsigned int offset = dist(gen);
										offset_list.push(offset);
										enqueue_dropout(
											*command_stream,
											random_uniform_buf,
											input_and_all_buffers_pack_it->first,
											dropout_it->second,
											mask,
											current_updater_entry_count * layer_config_it->get_neuron_count(),
											offset);
									}
								}

								(*it)->enqueue_test(
									(it == updater_list.begin()) ? base_input_entry_id : 0,
									*command_stream,
									*schema_data_it,
									*net_data_it,
									*net_data_custom_it,
									input_and_all_buffers_pack_it->first,
									input_and_all_buffers_pack_it->second.output_neurons_buffer,
									input_and_all_buffers_pack_it->second.additional_buffers,
									input_and_all_buffers_pack_it->second.dynamic_memobjects,
									current_updater_entry_count);
							}
						}

						// Compute errors
						{
							if (error_function_fused_with_activation)
								ef_updater->enqueue_update_error_and_gradient_fused_with_activation(
									*command_stream,
									initial_error_buf,
									error_buf,
									output_buf[current_command_slot],
									output_buffer,
									base_input_entry_id,
									output_neuron_count,
									current_updater_entry_count);
							else
								ef_updater->enqueue_update_error_and_gradient(
									*command_stream,
									initial_error_buf,
									error_buf,
									output_buf[current_command_slot],
									output_buffer,
									base_input_entry_id,
									output_neuron_count,
									current_updater_entry_count);
						}

						// Backward updater
						{
							std::vector<cuda_linear_buffer_device_smart_ptr>::iterator output_errors_it = output_errors_buffers.begin();
							std::vector<std::pair<cuda_linear_buffer_device_smart_ptr, layer_updater_cuda::buffer_set> >::reverse_iterator input_and_all_buffers_pack_it = updater_input_and_all_buffers_pack.rbegin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::reverse_iterator net_data_it = net_data.rbegin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::reverse_iterator net_data_custom_it = net_data_custom.rbegin();
							std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::reverse_iterator gradient_it = gradient.rbegin();
							std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::reverse_iterator schema_data_it = updater_schema_data.rbegin();
							unsigned int reverse_layer_id = static_cast<unsigned int>(updater_list.size() + testing_layer_count) - 1;
							layer_configuration_specific_list::const_reverse_iterator layer_config_it = layer_config_list.rbegin() + (1 + (error_function_fused_with_activation ? 1 : 0));
							for(std::vector<layer_updater_cuda_smart_ptr>::reverse_iterator it = updater_list.rbegin(); it != updater_list.rend(); ++it, ++input_and_all_buffers_pack_it, ++schema_data_it, ++gradient_it, ++output_errors_it, ++net_data_it, ++net_data_custom_it, --reverse_layer_id, ++layer_config_it)
							{
								(*it)->enqueue_update_weights(
									(it == (updater_list.rend() - 1)) ? base_input_entry_id : 0,
									*command_stream,
									*gradient_it,
									*net_data_custom_it,
									*schema_data_it,
									*output_errors_it,
									input_and_all_buffers_pack_it->first,
									input_and_all_buffers_pack_it->second.additional_buffers,
									input_and_all_buffers_pack_it->second.dynamic_memobjects,
									current_updater_entry_count);

								/*
								for(int part_id = 0; part_id < gradient_it->size(); ++part_id)
								{
									cuda_linear_buffer_device_smart_ptr buf = gradient_it->at(part_id);
									std::vector<float> inp_err(buf->get_size() / sizeof(float));
									cuda_safe_call(cudaMemcpyAsync(&(*inp_err.begin()), *buf, inp_err.size() * sizeof(float), cudaMemcpyDeviceToHost, *command_stream));
									cuda_safe_call(cudaStreamSynchronize(*command_stream));
										
									boost::filesystem::path dir = "Debug";
									dir /= "GPU";
									boost::filesystem::create_directories(dir);
									debug_util::dump_list(
										&(*inp_err.begin()),
										inp_err.size(),
										(dir / (boost::format("input_errors_%1%_%2%.txt") % reverse_layer_id % part_id).str()).string().c_str());
								}
								*/

								if (it != (updater_list.rend() - 1))
								{
									(*it)->enqueue_backprop(
										*command_stream,
										*schema_data_it,
										*net_data_it,
										*net_data_custom_it,
										input_and_all_buffers_pack_it->second.output_neurons_buffer,
										input_and_all_buffers_pack_it->first,
										*output_errors_it,
										input_and_all_buffers_pack_it->second.input_errors_buffer,
										input_and_all_buffers_pack_it->second.additional_buffers,
										input_and_all_buffers_pack_it->second.dynamic_memobjects,
										current_updater_entry_count);

									/*
									{
										cuda_linear_buffer_device_smart_ptr buf = (input_and_all_buffers_pack_it->second.input_errors_buffer == 0) ? *output_errors_it : input_and_all_buffers_pack_it->second.input_errors_buffer;
										std::vector<float> inp_err(buf->get_size() / sizeof(float));
										cuda_safe_call(cudaMemcpyAsync(&(*inp_err.begin()), *buf, inp_err.size() * sizeof(float), cudaMemcpyDeviceToHost, *command_stream));
										cuda_safe_call(cudaStreamSynchronize(*command_stream));
										
										boost::filesystem::path dir = "Debug";
										dir /= "GPU";
										boost::filesystem::create_directories(dir);
										debug_util::dump_list(
											&(*inp_err.begin()),
											inp_err.size(),
											(dir / (boost::format("input_errors_%1%.txt") % reverse_layer_id).str()).string().c_str());
									}
									*/

									std::map<unsigned int, float>::const_iterator dropout_it = layer_to_dropout_rate_map.find(reverse_layer_id);
									if (dropout_it != layer_to_dropout_rate_map.end())
									{
										unsigned int offset = offset_list.top();
										offset_list.pop();
										enqueue_dropout(
											*command_stream,
											random_uniform_buf,
											(input_and_all_buffers_pack_it->second.input_errors_buffer == 0) ? *output_errors_it : input_and_all_buffers_pack_it->second.input_errors_buffer,
											dropout_it->second,
											mask,
											current_updater_entry_count * layer_config_it->get_neuron_count(),
											offset);
									}
								}
							}
						}

						base_input_entry_id += current_updater_entry_count;
						entry_gradient_calculated_count += current_updater_entry_count;

						if (entry_gradient_calculated_count >= batch_size)
						{
							float gradient_normalizer = 1.0F / static_cast<float>(std::max(batch_size, entry_gradient_calculated_count));
							enqueue_apply_gradient(
								*command_stream,
								net_data,
								gradient,
								previous_upd,
								learning_rates,
								update_accum_buf,
								gradient_normalizer,
								weight_decay,
								momentum);
							entry_gradient_calculated_count = 0;
							++gradient_applied_count;
						}

						if (cuda_config->is_flush_required())
						{
							cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
							cuda_safe_call(cudaEventQuery(data_processed_event));
						}
					} // while(base_input_entry_id < entries_available_for_processing_count)

					entries_processed_count += entries_available_for_processing_count;

					if (cuda_config->is_flush_required())
					{
						cuda_safe_call(cudaEventRecord(data_processed_event, *command_stream));
						cuda_safe_call(cudaEventQuery(data_processed_event));
					}
				} // if (entries_available_for_processing_count > 0)

				unsigned int entries_read_count = 0;
				if (entries_available_for_copy_in_count > 0)
					entries_read_count = async_reader.wait();

				cuda_safe_call(cudaStreamSynchronize(*data_stream));
				cuda_safe_call(cudaStreamSynchronize(*command_stream));

				entries_available_for_processing_count = entries_read_count;
				entries_available_for_copy_in_count -= entries_read_count;

				current_data_slot = 1 - current_data_slot;
				current_command_slot = 1 - current_command_slot;
			}

			if (entry_gradient_calculated_count > 0)
			{
				float gradient_normalizer = 1.0F / static_cast<float>(std::max(batch_size, entry_gradient_calculated_count));
				enqueue_apply_gradient(
					*command_stream,
					net_data,
					gradient,
					previous_upd,
					learning_rates,
					update_accum_buf,
					gradient_normalizer,
					weight_decay,
					momentum);
				entry_gradient_calculated_count = 0;
				++gradient_applied_count;
			}

			read_data(net_data, data, *command_stream);

			double error;
			cuda_safe_call(cudaMemcpyAsync(&error, *error_buf, sizeof(double), cudaMemcpyDeviceToHost, *command_stream));
			cuda_safe_call(cudaStreamSynchronize(*command_stream));

			testing_res->init(error, entries_processed_count);

			training_stat_smart_ptr training_res = read_update_accum(
				update_accum_buf,
				data,
				gradient_applied_count,
				*command_stream);

			return std::make_pair(testing_res, training_res);
		}

		training_stat_smart_ptr network_updater_cuda::read_update_accum(
			const_cuda_linear_buffer_device_smart_ptr update_accum,
			network_data_smart_ptr data,
			unsigned int gradient_applied_count,
			cudaStream_t stream_id) const
		{
			training_stat_smart_ptr training_res(new training_stat());

			float mult = 1.0F / static_cast<float>(gradient_applied_count);

			std::vector<double> pack(update_accum->get_size() / sizeof(double));
			cuda_safe_call(cudaMemcpyAsync(&(*pack.begin()), *update_accum, update_accum->get_size(), cudaMemcpyDeviceToHost, stream_id));
			cuda_safe_call(cudaStreamSynchronize(stream_id));

			std::vector<double>::const_iterator current_accum_it = pack.begin();
			for(layer_data_list::const_iterator it = data->data_list.begin(); it != data->data_list.end(); ++it)
			{
				std::vector<float> layer_stat;

				for(std::vector<std::vector<float> >::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); ++it2)
				{
					int elem_count = it2->size();

					double sum = std::accumulate(
						current_accum_it, 
						current_accum_it + elem_count_update_accum_per_part,
						0.0);

					float val = static_cast<float>(sum) * mult / static_cast<float>(elem_count);
					layer_stat.push_back(val);

					current_accum_it += elem_count_update_accum_per_part;
				}

				training_res->absolute_updates.push_back(layer_stat);
			}

			return training_res;
		}

		void network_updater_cuda::layer_config_list_modified()
		{
			cuda_config->set_device();

			layer_configuration_specific_list::const_iterator it_conf = layer_config_list.begin();

			tester_list.clear();
			for(const_layer_testing_schema_list::const_iterator it = testing_schemas.begin(); it != testing_schemas.end(); ++it, ++it_conf)
			{
				tester_list.push_back(
					(*it)->create_tester(
						*it_conf,
						*(it_conf + 1)));
			}

			updater_list.clear();
			for(const_layer_updater_schema_list::const_iterator it = updater_schemas.begin(); it != updater_schemas.end(); ++it, ++it_conf)
			{
				updater_list.push_back(
					(*it)->create_updater(
						*it_conf,
						*(it_conf + 1),
						(it_conf > layer_config_list.begin() + testing_layer_count)));
			}
		}

		std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > network_updater_cuda::get_data(network_data_const_smart_ptr data) const
		{
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > res;

			for(int i = 0; i < updater_list.size(); ++i)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> device_data = updater_list[i]->get_data(data->data_list[i + testing_layer_count]);
				res.push_back(device_data);
			}

			return res;
		}

		std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > network_updater_cuda::set_get_data_custom(network_data_const_smart_ptr data)
		{
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > res;

			for(int i = 0; i < updater_list.size(); ++i)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> device_data = updater_list[i]->set_get_data_custom(data->data_custom_list[i + testing_layer_count]);
				res.push_back(device_data);
			}

			return res;
		}

		std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > network_updater_cuda::get_zero_gradient(const std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& data) const
		{
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> > res;

			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::const_iterator it = data.begin(); it != data.end(); ++it)
			{
				std::vector<cuda_linear_buffer_device_smart_ptr> device_data;
				for(std::vector<cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
				{
					size_t buf_size = (*it2)->get_size();
					cuda_linear_buffer_device_smart_ptr buf = cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(buf_size));
					cuda_util::set_with_value(
						*cuda_config,
						*buf,
						0.0F,
						buf_size / sizeof(float),
						0);
					device_data.push_back(buf);
				}
				res.push_back(device_data);
			}
			cuda_safe_call(cudaStreamSynchronize(0));

			return res;
		}

		void network_updater_cuda::read_data(
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& data_list,
			network_data_smart_ptr res,
			cudaStream_t stream_id) const
		{
			unsigned int layer_id = 0;
			for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator src_it = data_list.begin(); src_it != data_list.end(); ++src_it, ++layer_id)
				updater_list[layer_id]->get_data_from_device(*src_it, res->data_list[layer_id + testing_layer_count]);
		}

		void network_updater_cuda::update_buffers_configuration(
			buffer_cuda_size_configuration& buffer_configuration,
			unsigned int updater_entry_count) const
		{
			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = testing_schema_data.begin(); it != testing_schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_tester_cuda_smart_ptr>::const_iterator it = tester_list.begin(); it != tester_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration);

			for(std::vector<std::vector<const_cuda_linear_buffer_device_smart_ptr> >::const_iterator it = updater_schema_data.begin(); it != updater_schema_data.end(); ++it)
				for(std::vector<const_cuda_linear_buffer_device_smart_ptr>::const_iterator it2 = it->begin(); it2 != it->end(); ++it2)
					buffer_configuration.add_constant_buffer((*it2)->get_size());

			for(std::vector<layer_updater_cuda_smart_ptr>::const_iterator it = updater_list.begin(); it != updater_list.end(); ++it)
				(*it)->update_buffer_configuration(buffer_configuration, updater_entry_count);
		}

		void network_updater_cuda::enqueue_dropout(
			cudaStream_t stream_id,
			const_cuda_linear_buffer_device_smart_ptr random_buffer,
			cuda_linear_buffer_device_smart_ptr target_buffer,
			float dropout_rate,
			unsigned int mask,
			unsigned int elem_count,
			unsigned int offset_in_random_list)
		{
			float scale = 1.0F / (1.0F - dropout_rate);
			std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
				*cuda_config,
				elem_count);
			dropout_kernel<<<kernel_dims.first, kernel_dims.second, 0, stream_id>>>(
				*target_buffer,
				*random_buffer,
				dropout_rate,
				scale,
				offset_in_random_list,
				mask,
				elem_count);
		}

		void network_updater_cuda::enqueue_apply_gradient(
			cudaStream_t stream_id,
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& data,
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& gradient,
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >& prev_upd,
			const std::vector<std::vector<float> >& learning_rates,
			cuda_linear_buffer_device_smart_ptr update_accum,
			float gradient_normalizer,
			float weight_decay,
			float momentum)
		{
			const const_layer_list& layer_list = *schema;

			const_layer_list::const_iterator layer_it = layer_list.begin() + testing_layer_count;
			std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator gradient_it = gradient.begin();
			std::vector<std::vector<float> >::const_iterator learning_rate_it = learning_rates.begin() + testing_layer_count;
			unsigned int total_part_id = 0;
			if (momentum > 0.0F)
			{
				std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator prev_upd_it = prev_upd.begin();
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator data_it = data.begin(); data_it != data.end(); ++data_it, ++gradient_it, ++prev_upd_it, ++learning_rate_it, ++layer_it)
				{
					std::vector<cuda_linear_buffer_device_smart_ptr>::iterator gradient_it2 = gradient_it->begin();
					std::vector<cuda_linear_buffer_device_smart_ptr>::iterator prev_upd_it2 = prev_upd_it->begin();
					std::vector<float>::const_iterator learning_rate_it2 = learning_rate_it->begin();
					std::set<unsigned int> weight_decay_part_id_set = (*layer_it)->get_weight_decay_part_id_set();
					unsigned int part_id = 0;
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2, ++prev_upd_it2, ++learning_rate_it2, ++part_id)
					{
						float learning_rate = *learning_rate_it2;
						float actual_weight_decay = (weight_decay_part_id_set.find(part_id) == weight_decay_part_id_set.end()) ? 0.0F : weight_decay;
						int elem_count = (*data_it2)->get_size() / sizeof(float);
						std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
							*cuda_config,
							elem_count,
							1,
							1,
							32);
						int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
						int smem_size = threadblock_size * sizeof(float);
						apply_gradient_with_momentum_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
							**data_it2,
							**gradient_it2,
							**prev_upd_it2,
							((double *)(*update_accum)) + total_part_id * elem_count_update_accum_per_part,
							learning_rate,
							gradient_normalizer,
							actual_weight_decay,
							momentum,
							elem_count,
							elem_count_update_accum_per_part - 1);
						++total_part_id;
					}
				}
			}
			else
			{
				for(std::vector<std::vector<cuda_linear_buffer_device_smart_ptr> >::iterator data_it = data.begin(); data_it != data.end(); ++data_it, ++gradient_it, ++learning_rate_it, ++layer_it)
				{
					std::vector<cuda_linear_buffer_device_smart_ptr>::iterator gradient_it2 = gradient_it->begin();
					std::vector<float>::const_iterator learning_rate_it2 = learning_rate_it->begin();
					std::set<unsigned int> weight_decay_part_id_set = (*layer_it)->get_weight_decay_part_id_set();
					unsigned int part_id = 0;
					for(std::vector<cuda_linear_buffer_device_smart_ptr>::iterator data_it2 = data_it->begin(); data_it2 != data_it->end(); ++data_it2, ++gradient_it2, ++learning_rate_it2, ++part_id)
					{
						float learning_rate = *learning_rate_it2;
						float actual_weight_decay = (weight_decay_part_id_set.find(part_id) == weight_decay_part_id_set.end()) ? 0.0F : weight_decay;
						int elem_count = (*data_it2)->get_size() / sizeof(float);
						std::pair<dim3, dim3> kernel_dims = cuda_util::get_grid_and_threadblock_sizes_sequential_access(
							*cuda_config,
							elem_count,
							1,
							1,
							32);
						int threadblock_size = kernel_dims.second.x * kernel_dims.second.y * kernel_dims.second.z;
						int smem_size = threadblock_size * sizeof(float);
						apply_gradient_kernel<<<kernel_dims.first, kernel_dims.second, smem_size, stream_id>>>(
							**data_it2,
							**gradient_it2,
							((double *)(*update_accum)) + total_part_id * elem_count_update_accum_per_part,
							learning_rate,
							gradient_normalizer,
							actual_weight_decay,
							elem_count,
							elem_count_update_accum_per_part - 1);
						++total_part_id;
					}
				}
			}
		}
	}
}
