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
  if (textstream_ != NULL) delete textstream_;
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
  // first check if we have already a binary-csv
  binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin").c_str(),
    std::ios::in | std::ios::binary);
  bin_writing_ = false; // because you can't get the mode of opening of a stream
  LOG(INFO) << "Binstream opened with flag" << binstream_->rdstate();

  if(!binstream_->good()) {
    LOG(INFO) << "There is no BIN file yet, using CSV file";
    delete binstream_;
    binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin.part").c_str(),
      std::ios::out | std::ios::binary);
    bin_writing_ = true;
    textstream_ = new std::fstream(this->layer_param_.big_data_param().source().c_str(),
      std::ios::in);
    textstream_->seekg(0);
    file_smaller_than_chunk_ =
      (this->layer_param_.big_data_param().chunk_size() * 1000000) > (textstream_->tellg()/4);
    // why /4 you ask? We suppose file contains floats right? (<sep><num><.><num>)
    already_loaded_ = false;
    if (file_smaller_than_chunk_) {
      LOG(INFO) << "BigData loaded file is LESS than one chunk";
    } else {
      LOG(INFO) << "BigData loaded file is MORE than one chunk";
    }
    textstream_->seekg(0);
    delim_ = this->layer_param_.big_data_param().separator().c_str()[0];
    newline_ = this->layer_param_.big_data_param().newline().c_str()[0];

    // skip header if exists
    if(this->layer_param_.big_data_param().has_header()) {
      textstream_->ignore(2147483647, newline_);
    }
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

  LOG(INFO) << " TEXT, STARTS on pos: " << textstream_->tellg();

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
      if(has_label_) *binstream_ << *labels;
      // binstream_->write((char*) data, data_total * sizeof(*data));
      for(int i=0; i<data_total; ++i) *binstream_ << data[i];

      // move the pointers
      ++batch;
      data += data_total;
      if(has_label_) labels += 1;
    }

    // if we reached EOF || there was an empty line at the end of the file
    if(textstream_->eof() || data_col < data_total) {
      // why can't we just seekg(0)?! For some reason it doesnt' rewind the file :-/
      LOG(INFO) << "reset file at batch " << batch;
      binstream_->close();
      textstream_->close();
      delete textstream_;
      delete binstream_;
      textstream_ = NULL;
      std::rename( // strip the ".part" extension when the file is done
        string(this->layer_param_.big_data_param().source() + "bin.part").c_str(),
        string(this->layer_param_.big_data_param().source() + "bin").c_str());
      // reuse the binary file
      binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin").c_str(),
        std::ios::in | std::ios::binary);
      bin_writing_ = false;
      // means there was an empty line at the end of the file, so restart loading
      return ReadFromBin(how_many - batch, data, labels);
    }
    LOG(INFO) << " CSV " << batch << "/" << how_many
      << " D: " << data[-data_total] << "..." << data[-1]
      << " L: " << labels[-1]; // TODO: dangerous, remove
  }
}


template <typename Dtype>
void BigDataLayer<Dtype>::ReadFromBin(size_t how_many, Dtype* data, Dtype* labels)
{
  LOG(INFO) << " BIN,  STARTS on pos: " << binstream_->tellg();
  size_t data_total = (shape_[1] * shape_[2] * shape_[3]);

  for(int batch=0; batch < how_many; ++batch)
  {
    if(has_label_) {
      *binstream_ >> *labels;
      labels += 1;
    }
    // binstream_->read((char*)data, data_total * sizeof(*data));
    for(int i=0; i<data_total; ++i) *binstream_ >> data[i];
    data += data_total;
    if(binstream_->eof()) {
      binstream_->close();
      delete binstream_;
      binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin").c_str(),
        std::ios::in | std::ios::binary);
    }
    LOG(INFO) << " BIN " << batch << "/" << how_many
    << " D: " << data[-data_total] << "..." << data[-1]
    << " L: " << labels[-1]; // TODO: dangerous, remove
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
    LOG(INFO) << "File smaller than batch -- using last loaded data";
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

  if(bin_writing_)
    ReadFromText(shape_[0], top_data, top_label);
  else
    ReadFromBin(shape_[0], top_data, top_label);

  already_loaded_ = true;
  // LOG(INFO) << "BigData: thread #" << " ENDS on file pos: " << istream->tellg();

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(BigDataLayer);
REGISTER_LAYER_CLASS(BigData);

}  // namespace caffe
