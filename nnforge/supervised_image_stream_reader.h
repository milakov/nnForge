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

#pragma once

#include "supervised_data_reader.h"

#include <vector>
#include <istream>
#include <opencv2/core/core.hpp>
#include <boost/thread/thread.hpp>

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif 

#include <boost/asio/io_service.hpp>

namespace nnforge
{
	class supervised_image_stream_reader : public supervised_data_reader
	{
	public:
		// The constructor modifies input_stream to throw exceptions in case of failure
		supervised_image_stream_reader(
			nnforge_shared_ptr<std::istream> input_stream,
			unsigned int target_image_width,
			unsigned int target_image_height,
			bool fit_into_target,
			bool is_color = true,
			unsigned int prefetch_count = 3);

		virtual ~supervised_image_stream_reader();

		virtual void reset();

		virtual bool raw_read(std::vector<unsigned char>& all_elems);

		virtual neuron_data_type::input_type get_input_type() const
		{
			return neuron_data_type::type_byte;
		}

		virtual void rewind(unsigned int entry_id);

		virtual unsigned int get_entry_count() const
		{
			return static_cast<unsigned int>(entry_offsets.size() - 1);
		}

	protected:
		bool entry_available();

		virtual bool read_image(
			cv::Mat * image,
			unsigned int * class_id);

	protected:
		class request
		{
		public:
			virtual ~request()
			{
			}
		};

		class read_request : public request
		{
		public:
			read_request(unsigned int entry_id, cv::Mat * image, unsigned int * class_id)
				: entry_id(entry_id)
				, image(image)
				, class_id(class_id)
			{
			};

			unsigned int entry_id;
			cv::Mat * image;
			unsigned int * class_id;
		};

		class raw_read_request : public request
		{
		public:
			raw_read_request(unsigned int entry_id, std::vector<unsigned char> * all_elems)
				: entry_id(entry_id)
				, all_elems(all_elems)
			{
			};

			unsigned int entry_id;
			std::vector<unsigned char> * all_elems;
		};

		class decode_data_info
		{
		public:
			decode_data_info(unsigned int class_id);

			cv::Mat image;
			unsigned int class_id;
			bool is_ready;
			boost::mutex is_ready_mutex;
			boost::condition_variable is_ready_condition;
			std::string decode_worker_error;
		};

	protected:
		nnforge_shared_ptr<std::istream> in_stream;
		unsigned int target_image_width;
		unsigned int target_image_height;
		bool fit_into_target;
		bool is_color;
		unsigned int prefetch_count;

		std::vector<unsigned long long> entry_offsets;
		unsigned int entry_read_count;
		std::istream::pos_type reset_pos;
		std::vector<unsigned char> buf;

		nnforge_shared_ptr<boost::thread> prefetch_thread;
		boost::mutex request_pending_mutex;
		boost::condition_variable prefetch_request_pending_condition;
		bool prefetch_stop_requested;
		bool prefetch_request_ready;
		boost::mutex request_processed_mutex;
		boost::condition_variable prefetch_request_processed_condition;
		bool prefetch_request_processed;
		nnforge_shared_ptr<request> current_request;
		std::string prefetch_worker_error;

		std::map<unsigned int, nnforge_shared_ptr<decode_data_info> > decode_cache_map;

	private:
		supervised_image_stream_reader(const supervised_image_stream_reader&);
		supervised_image_stream_reader& operator =(const supervised_image_stream_reader&);

		void start_prefetch();

		void read(
			unsigned int entry_id,
			cv::Mat * image,
			unsigned int * class_id);

		void raw_read(
			unsigned int entry_id,
			std::vector<unsigned char>& all_elems);

		void read_data_from_cache(
			unsigned int entry_id,
			cv::Mat * image,
			unsigned int * class_id);

		void start_prefetch(
			boost::asio::io_service& io_service,
			unsigned int entry_id);

		static void prefetch_worker(
			supervised_image_stream_reader * reader,
			nnforge_shared_ptr<std::vector<unsigned char> > raw_input_data,
			nnforge_shared_ptr<decode_data_info> decode_info);

		void decode(
			const std::vector<unsigned char>& raw_data,
			cv::Mat& image) const;

		struct prefetch_thread_info
		{
			 void operator()();
			 supervised_image_stream_reader * reader;
		};
	};
}
