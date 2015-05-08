#include <stdint.h>
#include <cstdio> // rename(char*, char*)

#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <boost/thread.hpp>
// #include <boost/filesystem/operations.hpp>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
BigDataLayer<Dtype>::~BigDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  if (textstream_ != NULL) {
    if(textstream_->is_open()) textstream_->close();
    delete textstream_;
  }
  if (binstream_ != NULL) {
    if(binstream_->is_open()) binstream_->close();
    delete binstream_;
  }
}

template <typename Dtype>
void BigDataLayer<Dtype>::ResetStream(std::fstream* newstream) {
  this->JoinPrefetchThread();
  LOG(WARNING) << "BigData stream reset called";
  if(textstream_ != NULL) delete textstream_;
  textstream_ = newstream;
  this->CreatePrefetchThread();
}

template <typename Dtype>
void BigDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CPUTimer init_timer;
  init_timer.Start();
  file_smaller_than_chunk_ = false;
  already_loaded_ = false;
  // first check if we have already a binary-csv
  binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin").c_str(),
    std::ios::in | std::ios::binary);
  has_binary_ = binstream_->good();

  if(binstream_->good())
  {
    // find out if the data will fit in one batch (so we can cache them)
    binstream_->seekg(0, std::ios_base::end);
    file_smaller_than_chunk_ =
      (this->layer_param_.big_data_param().chunk_size() * 1000000) > (binstream_->tellg());
    binstream_->seekg(0);
    LOG(INFO) << "Using converted BINARY CSV file instead the original one!";
  }
  else
  {
    textstream_ = new std::fstream(this->layer_param_.big_data_param().source().c_str(),
      std::ios::in);
    // find out if the data will fit in one batch (so we can cache them)
    textstream_->seekg(0, std::ios_base::end);
    file_smaller_than_chunk_ =
      (this->layer_param_.big_data_param().chunk_size() * 1000000) > (textstream_->tellg()/2);
    textstream_->seekg(0);

    delim_ = this->layer_param_.big_data_param().separator().c_str()[0];
    newline_ = this->layer_param_.big_data_param().newline().c_str()[0];

    // skip # of lines denoted to header
    for(int i=0; i < this->layer_param_.big_data_param().header(); ++i) {
      textstream_->ignore(2147483647, newline_);
    }

    // init binary stream for writing
    delete binstream_;
    binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin.part").c_str(),
      std::ios::out | std::ios::binary);

  }

  if (file_smaller_than_chunk_) {
    LOG(INFO) << "Source file is LESS than one chunk - will use memory caching";
  }

  // save indices describing data
  data_start_ = this->layer_param_.big_data_param().data_start();
  data_end_ = this->layer_param_.big_data_param().data_end();
  label_ = this->layer_param_.big_data_param().label();

  shape_ = vector<int>(4);
  // if the user has specified shape that means that his data are multi-dimensional
  // if(this->layer_param_.big_data_param().has_shape()) {
  //   shape_[3] = this->layer_param_.big_data_param().shape().width();
  //   shape_[2] = this->layer_param_.big_data_param().shape().height();
  //   shape_[1] = this->layer_param_.big_data_param().shape().channel();
  // } else {
    // otherwise we axpect one dimensional data
    shape_[3] = data_end_ - data_start_ + 1;
    shape_[2] = 1;
    shape_[1] = 1;
  // }
  // the maximal batch size is computed from chunk_size (which is in MB)
  shape_[0] = ceil((1000000 / (sizeof(Dtype) * shape_[3] * shape_[2] * shape_[1])) *
                   this->layer_param_.big_data_param().chunk_size());
  top[0]->Reshape(shape_);
  this->prefetch_data_.Reshape(shape_);

  // Read a data point, and use it to initialize the top blob.
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  has_label_ = this->layer_param_.big_data_param().has_label();
  if (has_label_) {
    label_shape_ = vector<int>(1, shape_[0]);
    top[1]->Reshape(label_shape_);
    this->prefetch_label_.Reshape(label_shape_);
  }
  init_timer.Stop();
  DLOG(INFO) << "Init BigDataLayer time: " << init_timer.MilliSeconds() << " ms.";
}


template <typename Dtype>
void BigDataLayer<Dtype>::ReadFromText(size_t how_many, Dtype* data, Dtype* labels)
{
  char buff[255];
  size_t data_total = (shape_[1] * shape_[2] * shape_[3]);
  size_t batch = 0, current_col = 0, data_col = 0;
  bool got_label = false;

  // LOG(INFO) << " TEXT, STARTS on pos: " << textstream_->tellg();

  while(batch < how_many)
  {
    data_col = 0;
    current_col = 0;
    got_label = false;
    // read a row from text file
    while(textstream_->good() && (has_label_ != got_label || data_col < data_total))
    {
      // linestream.getline(buff, 255, delim_);
      textstream_->getline(buff, 255, delim_);
      if(current_col >= data_start_ && current_col <= data_end_) {
        data[data_col] = atof(buff);
        ++data_col;
      }

      if(current_col == label_) {
        *labels = atof(buff);
        got_label = true;
      }

      ++current_col;
    }
    textstream_->ignore(2147483647, newline_);

    if (data_col == data_total) {
      // save the row to the binary file
      if(has_label_) binstream_->write((char*)labels, sizeof(*labels));
      binstream_->write((char*) data, data_total * sizeof(*data));
      // move the pointers
      ++batch;
      data += data_total;
      if(has_label_) labels += 1;
    }

    // if we reached EOF || there was an empty line at the end of the file
    if(textstream_->eof() || data_col < data_total) {
      // LOG(INFO) << "Reset file at batch " << batch;
      binstream_->close();
      textstream_->close();
      delete textstream_;
      textstream_ = NULL;
      std::rename( // strip the ".part" extension when the file is done
        string(this->layer_param_.big_data_param().source() + "bin.part").c_str(),
        string(this->layer_param_.big_data_param().source() + "bin").c_str());
      // reuse the binary file
      delete binstream_;
      binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin").c_str(),
        std::ios::in | std::ios::binary);
      has_binary_ = true;
      // means there was an empty line at the end of the file, so restart loading
      return ReadFromBin(how_many - batch, data, labels);
    }
    // LOG(INFO) << " CSV " << batch << "/" << how_many
    //   << " D: " << data[-data_total] << "..." << data[-1]
    //   << " L: " << labels[-1]; // TODO: dangerous, remove
  }
}


template <typename Dtype>
void BigDataLayer<Dtype>::ReadFromBin(size_t how_many, Dtype* data, Dtype* labels)
{
  // LOG(INFO) << " BIN,  STARTS on pos: " << binstream_->tellg();
  size_t data_total = (shape_[1] * shape_[2] * shape_[3]);

  for(int batch=0; batch < how_many; ++batch)
  {
    if(has_label_) {
      binstream_->read((char*)labels, sizeof(*labels));
      labels += 1;
    }
    binstream_->read((char*)data, data_total * sizeof(*data));
    data += data_total;
    if(binstream_->eof()) {
      // this flags get set in case of unsucessful read -- means that we should
      // check it always after the first read and then re-read ... here we leave
      // one old datum in the prefetched_data ... which doesn't hurt, does it?
      binstream_->close();
      delete binstream_;
      binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin").c_str(),
        std::ios::in | std::ios::binary);
    }
    // LOG(INFO) << " BIN " << batch << "/" << how_many
    // << " D: " << data[-data_total] << "..." << data[-1]
    // << " L: " << labels[-1]; // TODO: dangerous, remove
  }
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void BigDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();

  // if we have read the whole file and it's smaller than chunk, don't read it again
  if(file_smaller_than_chunk_ && already_loaded_) {
    batch_timer.Stop();
    // LOG(INFO) << "File smaller than batch -- using last loaded data";
    return;
  }

  // revert prefetched data and labels back to it's full size
  this->prefetch_data_.Reshape(shape_);
  if(has_label_) this->prefetch_label_.Reshape(label_shape_);

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if(has_label_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }

  if(has_binary_)
    ReadFromBin(shape_[0], top_data, top_label);
  else
    ReadFromText(shape_[0], top_data, top_label);

  already_loaded_ = true;
  // LOG(INFO) << "BigData: thread #" << " ENDS on file pos: " << istream->tellg();

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(BigDataLayer);
REGISTER_LAYER_CLASS(BigData);

}  // namespace caffe
