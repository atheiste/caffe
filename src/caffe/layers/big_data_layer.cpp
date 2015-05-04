#include <stdint.h>

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
  if (istream_ != NULL) delete istream_;
}

template <typename Dtype>
void BigDataLayer<Dtype>::ResetStream(std::istream* newstream) {
  this->JoinPrefetchThread();
  LOG(WARNING) << "BigData stream reset called";
  if(istream_ != NULL) delete istream_;
  istream_ = newstream;
  this->CreatePrefetchThread();
}

template <typename Dtype>
void BigDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CPUTimer init_timer;
  init_timer.Start();
  // setup everything with reading from file
  istream_ = new std::ifstream(this->layer_param_.big_data_param().source().c_str());
  istream_->seekg(0, std::ios::end);
  file_smaller_than_chunk_ =
    (this->layer_param_.big_data_param().chunk_size() * 1000000) > (istream_->tellg()/4);
  // why /4 you ask? We suppose file contains floats right? (<sep><num><.><num>)
  already_loaded_ = false;
  if (file_smaller_than_chunk_) {
    LOG(INFO) << "BigData loaded file is LESS than one chunk";
  } else {
    LOG(INFO) << "BigData loaded file is MORE than one chunk";
  }
  istream_->seekg(0);
  delim_ = this->layer_param_.big_data_param().separator().c_str()[0];
  newline_ = this->layer_param_.big_data_param().newline().c_str()[0];

  // skip header if exists
  if(this->layer_param_.big_data_param().has_header()) {
    istream_->ignore(2147483647, newline_);
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

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void BigDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  // CPUTimer timer;
  // batch_timer.Start();
  // double read_time = 0;
  // double trans_time = 0;
  std::ifstream *istream = dynamic_cast<std::ifstream*>(istream_);
  std::istringstream linestream;
  std::string line;
  char buff[255];

  // LOG(INFO) << "BigData: thread #" << boost::this_thread::get_id() << " STARTS on file pos: " << istream->tellg();

  // if we have read the whole file and it's smaller than chunk, don't read it again
  if(file_smaller_than_chunk_ && already_loaded_)
  {
    batch_timer.Stop();
    // LOG(INFO) << "File smaller than batch -- using last loaded data";
    return;
  }

  // revert prefetched data and labels back to it's full size
  this->prefetch_data_.Reshape(shape_);
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* cur_data = top_data;
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* cur_label = NULL;

  if(has_label_) {
    this->prefetch_label_.Reshape(label_shape_);
    top_label = this->prefetch_label_.mutable_cpu_data();
    cur_label = top_label;
  }

  size_t data_total = (shape_[1] * shape_[2] * shape_[3]);
  size_t current_batch = 0, current_col = 0, data_col = 0;
  bool got_label = false;

  // parse until we fill a whole chunk (or get interrupted)
  while(current_batch < shape_[0])
  {
    current_col = 0;
    data_col = 0;
    std::getline(*istream, line);
    linestream.str(line);
    linestream.clear(); // clear any bad flags

    while(linestream.good() && (has_label_ != got_label || data_col < data_total))
    {
      linestream.getline(buff, 255, delim_);

      if(current_col >= data_start_ && current_col <= data_end_) {
        cur_data[data_col] = atof(buff);
        ++data_col;
      }

      if(current_col == label_) {
        *cur_label = atof(buff);
        got_label = true;
      }

      ++current_col;
    }

    // rewinding file clears eof flag
    if(istream->eof()) {
      // why can't we just seekg(0)?! For some reason it doesnt' rewind the file :-/
      // LOG(INFO) << "BigData reset file at batch " << current_batch;
      istream->close();
      delete istream_;
      istream = new std::ifstream(this->layer_param_.big_data_param().source().c_str());
      istream_ = istream;
      if (data_col < data_total) {
        // means there was an empty line at the end of the file, so restart loading
        continue;
      }
    }

    cur_label += 1;
    cur_data += data_col;

    ++current_batch;
    // LOG(INFO) << "BigData read " << current_batch << "/" << shape_[0]
    //   << "; sneak peak of data: " << cur_data[-data_col] << "..." << cur_data[-1]
    //   << " label: " << cur_label[-1]; // TODO: dangerous, remove

    // istream->ignore(2147483647, newline_);

    if(boost::this_thread::interruption_requested()) {
      LOG(INFO) << "BigData got interrupt request at " << current_batch;
      // don't return smaller data than 1/10th of the optimal batch size
      // or don't quit if the whole file is smaller than one chunk
      if(current_batch < int(shape_[0]/10) || file_smaller_than_chunk_) continue;
      // otherwise break out
      break;
    }
  }

  if(current_batch < shape_[0]) {
    // we were interrupted or the file ended
    this->prefetch_data_.Reshape(current_batch, shape_[1], shape_[2], shape_[3]);
    if(has_label_) this->prefetch_label_.Reshape(current_batch,1,1,1);
  }

  already_loaded_ = true;
  // LOG(INFO) << "BigData: thread #" << boost::this_thread::get_id() << " ENDS on file pos: " << istream->tellg();

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  // DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  // DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(BigDataLayer);
REGISTER_LAYER_CLASS(BigData);

}  // namespace caffe
