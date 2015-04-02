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

#include "supervised_image_stream_reader.h"

#include <boost/format.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/bind.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <utility>
#include <algorithm>

#include "varying_data_stream_schema.h"
#include "neural_network_exception.h"

namespace nnforge
{
	supervised_image_stream_reader::supervised_image_stream_reader(
		nnforge_shared_ptr<std::istream> input_stream,
		unsigned int target_image_width,
		unsigned int target_image_height,
		bool fit_into_target,
		bool dynamic_output,
		bool is_color,
		unsigned int prefetch_count)
		: in_stream(input_stream)
		, target_image_width(target_image_width)
		, target_image_height(target_image_height)
		, fit_into_target(fit_into_target)
		, dynamic_output(dynamic_output)
		, is_color(is_color)
		, prefetch_count(prefetch_count)
		, entry_read_count(0)
	{
		in_stream->exceptions(std::ostream::eofbit | std::ostream::failbit | std::ostream::badbit);

		boost::uuids::uuid guid_read;
		in_stream->read(reinterpret_cast<char*>(guid_read.data), sizeof(guid_read.data));
		if (guid_read != varying_data_stream_schema::varying_data_stream_guid)
			throw neural_network_exception((boost::format("Unknown varying data GUID encountered in input stream: %1%") % guid_read).str());

		unsigned int entry_count;
		in_stream->read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));
		entry_offsets.resize(entry_count + 1);

		in_stream->read(reinterpret_cast<char*>(&(*entry_offsets.begin())), sizeof(unsigned long long) * entry_offsets.size());

		reset_pos = in_stream->tellg();

		prefetch_thread_info thread_info;
		thread_info.reader = this;
		prefetch_stop_requested = false;
		prefetch_request_ready = false;
		prefetch_thread = nnforge_shared_ptr<boost::thread>(new boost::thread(thread_info));
	}

	supervised_image_stream_reader::~supervised_image_stream_reader()
	{
		{
			boost::lock_guard<boost::mutex> lock(request_pending_mutex);
			prefetch_request_ready = true;
			prefetch_stop_requested = true;
		}
		prefetch_request_pending_condition.notify_one();
		if (prefetch_thread)
			prefetch_thread->join();
	}

	void supervised_image_stream_reader::prefetch_thread_info::operator()()
	{
		this->reader->start_prefetch();
	}

	void supervised_image_stream_reader::reset()
	{
		rewind(0);
	}

	void supervised_image_stream_reader::read(
		unsigned int entry_id,
		cv::Mat * image,
		std::vector<unsigned char>* output_data)
	{
		in_stream->seekg(reset_pos + (std::istream::off_type)(entry_offsets[entry_id]), std::ios::beg);

		unsigned long long total_entry_size = entry_offsets[entry_id + 1] - entry_offsets[entry_id];
		unsigned int input_data_size;
		if (dynamic_output)
			in_stream->read(reinterpret_cast<char*>(&input_data_size), sizeof(unsigned int));
		else
			input_data_size = static_cast<unsigned int>(total_entry_size - sizeof(unsigned int));

		if (image)
		{
			buf.resize(input_data_size);
			in_stream->read(reinterpret_cast<char*>(&(*buf.begin())), input_data_size);

			decode(buf, *image);
		}
		else
			in_stream->seekg(input_data_size, std::ios_base::cur);

		unsigned int output_data_size;
		if (dynamic_output)
			output_data_size = static_cast<unsigned int>(total_entry_size - input_data_size - sizeof(unsigned int));
		else
			output_data_size = sizeof(unsigned int);
		if (output_data)
		{
			output_data->resize(output_data_size);
			if (output_data_size > 0)
				in_stream->read(reinterpret_cast<char*>(&(output_data->at(0))), output_data_size);
		}
		else
			in_stream->seekg(output_data_size, std::ios_base::cur);
	}

	bool supervised_image_stream_reader::read_image(
		cv::Mat * image,
		std::vector<unsigned char> * output_data)
	{
		if (!entry_available())
			return false;

		// Set command
		current_request = nnforge_shared_ptr<request>(new read_request(entry_read_count, image, output_data));
		prefetch_request_processed = false;
		{
			boost::lock_guard<boost::mutex> lock(request_pending_mutex);
			prefetch_request_ready = true;
		}
		prefetch_request_pending_condition.notify_one();

		// Wait for the request to be fully processed
		{
			boost::unique_lock<boost::mutex> lock(request_processed_mutex);
			while (!prefetch_request_processed)
				prefetch_request_processed_condition.wait(lock);
		}

		if (!prefetch_worker_error.empty())
			throw std::runtime_error(prefetch_worker_error);

		entry_read_count++;

		return true;
	}

	void supervised_image_stream_reader::raw_read(
		unsigned int entry_id,
		std::vector<unsigned char>& all_elems)
	{
		in_stream->seekg(reset_pos + (std::istream::off_type)(entry_offsets[entry_id]), std::ios::beg);

		unsigned long long total_entry_size = entry_offsets[entry_id + 1] - entry_offsets[entry_id];
		all_elems.resize(total_entry_size);
		in_stream->read(reinterpret_cast<char*>(&(*all_elems.begin())), total_entry_size);
	}

	bool supervised_image_stream_reader::raw_read(std::vector<unsigned char>& all_elems)
	{
		if (!entry_available())
			return false;

		// Set command
		current_request = nnforge_shared_ptr<request>(new raw_read_request(entry_read_count, &all_elems));
		prefetch_request_processed = false;
		{
			boost::lock_guard<boost::mutex> lock(request_pending_mutex);
			prefetch_request_ready = true;
		}
		prefetch_request_pending_condition.notify_one();

		// Wait for the request to be fully processed
		{
			boost::unique_lock<boost::mutex> lock(request_processed_mutex);
			while (!prefetch_request_processed)
				prefetch_request_processed_condition.wait(lock);
		}

		if (!prefetch_worker_error.empty())
			throw std::runtime_error(prefetch_worker_error);

		entry_read_count++;

		return true;
	}

	bool supervised_image_stream_reader::entry_available()
	{
		return (entry_read_count < entry_offsets.size() - 1);
	}

	void supervised_image_stream_reader::rewind(unsigned int entry_id)
	{
		entry_read_count = entry_id;
	}

	void supervised_image_stream_reader::start_prefetch()
	{
		try
		{
			boost::asio::io_service io_service;
			boost::thread_group threadpool;
			boost::asio::io_service::work work(io_service);
			for(unsigned int i = 0; i < prefetch_count; ++i)
				threadpool.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));

			boost::unique_lock<boost::mutex> lock(request_pending_mutex);
			while(true)
			{
				while (!prefetch_request_ready)
					prefetch_request_pending_condition.wait(lock);

				prefetch_request_ready = false;

				if (prefetch_stop_requested)
					break;

				bool run_prefetch = false;

				// Process request
				nnforge_shared_ptr<read_request> read_request_derived = nnforge_dynamic_pointer_cast<read_request>(current_request);
				if (read_request_derived)
				{
					if (read_request_derived->image)
					{
						read_data_from_cache(read_request_derived->entry_id, read_request_derived->image, read_request_derived->output_data);
						run_prefetch = true;
					}
					else
					{
						read(read_request_derived->entry_id, read_request_derived->image, read_request_derived->output_data);
					}
				}
				else
				{
					nnforge_shared_ptr<raw_read_request> raw_read_request_derived = nnforge_dynamic_pointer_cast<raw_read_request>(current_request);
					if (raw_read_request_derived)
					{
						raw_read(raw_read_request_derived->entry_id, *raw_read_request_derived->all_elems);
					}
					else
						throw neural_network_exception("Unknown request specified for supervised_image_data_stream_reader worker");
				}

				// Notify caller thread that result is ready
				{
					boost::lock_guard<boost::mutex> lock(request_processed_mutex);
					prefetch_request_processed = true;
				}
				prefetch_request_processed_condition.notify_one();

				if (run_prefetch)
				{
					unsigned int start_prefetch_entry_id = read_request_derived->entry_id + 1;
					unsigned int end_prefetch_entry_id = std::min(start_prefetch_entry_id + prefetch_count, static_cast<unsigned int>(entry_offsets.size() - 1));

					std::map<unsigned int, nnforge_shared_ptr<decode_data_info> >::iterator current_it = decode_cache_map.begin();
					while (current_it != decode_cache_map.end())
					{
						if ((current_it->first < start_prefetch_entry_id) || (current_it->first >= end_prefetch_entry_id))
							decode_cache_map.erase(current_it++);
						else
							++current_it;
					}

					for(unsigned int prefetch_entry_id = start_prefetch_entry_id; prefetch_entry_id < end_prefetch_entry_id; ++prefetch_entry_id)
					{
						if (decode_cache_map.find(prefetch_entry_id) == decode_cache_map.end())
							start_prefetch(io_service, prefetch_entry_id);
					}
				}
			}

			io_service.stop();
			threadpool.join_all();
		}
		catch (std::runtime_error& e)
		{
			prefetch_worker_error = e.what();

			{
				boost::lock_guard<boost::mutex> lock(request_processed_mutex);
				prefetch_request_processed = true;
			}
			prefetch_request_processed_condition.notify_one();
		}
	}

	void supervised_image_stream_reader::read_data_from_cache(
		unsigned int entry_id,
		cv::Mat * image,
		std::vector<unsigned char> * output_data)
	{
		std::map<unsigned int, nnforge_shared_ptr<decode_data_info> >::iterator it = decode_cache_map.find(entry_id);
		if (it == decode_cache_map.end())
		{
			// Not in cache
			read(entry_id, image, output_data);
		}
		else
		{
			nnforge_shared_ptr<decode_data_info> info = it->second;

			// In cache but might not yet finished yet
			boost::unique_lock<boost::mutex> lock(info->is_ready_mutex);
			while (!info->is_ready)
				info->is_ready_condition.wait(lock);

			if (!info->decode_worker_error.empty())
				throw std::runtime_error((boost::format("Error when decoding entry %1%: %2%") % entry_id % info->decode_worker_error).str());

			*image = info->image;
			if (output_data)
				*output_data = info->output_data;
		}
	}

	void supervised_image_stream_reader::prefetch_worker(
		supervised_image_stream_reader * reader,
		nnforge_shared_ptr<std::vector<unsigned char> > raw_input_data,
		nnforge_shared_ptr<decode_data_info> decode_info)
	{
		try
		{
			reader->decode(*raw_input_data, decode_info->image);
			{
				boost::lock_guard<boost::mutex> lock(decode_info->is_ready_mutex);
				decode_info->is_ready = true;
			}
			decode_info->is_ready_condition.notify_one();
		}
		catch (std::runtime_error& e)
		{
			decode_info->decode_worker_error = e.what();
			{
				boost::lock_guard<boost::mutex> lock(decode_info->is_ready_mutex);
				decode_info->is_ready = true;
			}
			decode_info->is_ready_condition.notify_one();
		}
	}

	void supervised_image_stream_reader::start_prefetch(
		boost::asio::io_service& io_service,
		unsigned int entry_id)
	{
		in_stream->seekg(reset_pos + (std::istream::off_type)(entry_offsets[entry_id]), std::ios::beg);

		unsigned long long total_entry_size = entry_offsets[entry_id + 1] - entry_offsets[entry_id];
		unsigned int input_data_size;
		if (dynamic_output)
			in_stream->read(reinterpret_cast<char*>(&input_data_size), sizeof(unsigned int));
		else
			input_data_size = static_cast<unsigned int>(total_entry_size - sizeof(unsigned int));

		nnforge_shared_ptr<std::vector<unsigned char> > raw_input_data(new std::vector<unsigned char>(input_data_size));
		in_stream->read(reinterpret_cast<char*>(&(*raw_input_data->begin())), input_data_size);

		unsigned int output_data_size;
		if (dynamic_output)
			output_data_size = static_cast<unsigned int>(total_entry_size - input_data_size - sizeof(unsigned int));
		else
			output_data_size = sizeof(unsigned int);
		std::vector<unsigned char> output_data(output_data_size);
		if (output_data_size > 0)
			in_stream->read(reinterpret_cast<char*>(&(output_data[0])), output_data_size);

		nnforge_shared_ptr<decode_data_info> decode_info(new decode_data_info(output_data));
		decode_cache_map.insert(std::make_pair(entry_id, decode_info));

		io_service.post(boost::bind(prefetch_worker, this, raw_input_data, decode_info));
	}

	void supervised_image_stream_reader::decode(
		const std::vector<unsigned char>& raw_data,
		cv::Mat& image) const
	{
		cv::Mat original_image = cv::imdecode(cv::InputArray(raw_data), is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

		float width_ratio = static_cast<float>(target_image_width) / static_cast<float>(original_image.cols);
		float height_ratio = static_cast<float>(target_image_height) / static_cast<float>(original_image.rows);

		unsigned int dest_image_width;
		unsigned int dest_image_height;

		if ((width_ratio > height_ratio) ^ fit_into_target)
		{
			dest_image_width = target_image_width;
			dest_image_height = static_cast<unsigned int>(original_image.rows * width_ratio + 0.5F);
		}
		else
		{
			dest_image_width = static_cast<unsigned int>(original_image.cols * height_ratio + 0.5F);
			dest_image_height = target_image_height;
		}

		image = cv::Mat(dest_image_height, dest_image_width, original_image.type());

		cv::resize(original_image, image, cv::Size(image.cols, image.rows), 0.0, 0.0, CV_INTER_AREA);
	}

	supervised_image_stream_reader::decode_data_info::decode_data_info(const std::vector<unsigned char>& output_data)
		: output_data(output_data)
		, is_ready(false)
	{
	}
}
