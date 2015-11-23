#include <iostream>
#include <string>
#include <vector>
#include <cstdio>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename Dtype>
class BigDataLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  BigDataLayerTest()
      : filename_("test/test_data/bigdata.csv"),
        cols_(5), rows_(2)
  {
    top_data_ = new Blob<Dtype>();
    top_label_ = new Blob<Dtype>();
    top_ids_ = new Blob<Dtype>();

    top_vec_.push_back(top_data_);
    top_vec_.push_back(top_label_);
    top_vec_.push_back(top_ids_);
  }

  virtual void SetUp()
  {
    layer_param_.set_phase(::caffe::TRAIN);
    big_data_param_ = layer_param_.mutable_big_data_param();
    big_data_param_->set_chunk_size(0.05);
    big_data_param_->set_source(this->filename_.c_str());
    big_data_param_->set_header(1);
    big_data_param_->set_data_start(1);
    big_data_param_->set_data_end(5);
    big_data_param_->set_label(6);
    big_data_param_->set_cache(::caffe::BigDataParameter_CacheControl_DISABLED);
  }

  virtual ~BigDataLayerTest() {
    delete top_data_;
    delete top_label_;
    delete top_ids_;
  }

  const string filename_;
  const size_t cols_, rows_;
  Blob<Dtype> *top_data_, *top_label_, *top_ids_;
  vector<Blob<Dtype>*> bottom_vec_, top_vec_;
  LayerParameter layer_param_;
  ::caffe::BigDataParameter* big_data_param_;
};

TYPED_TEST_CASE(BigDataLayerTest, TestDtypes);

TYPED_TEST(BigDataLayerTest, TestRead) {
  // please update when updating test file
  TypeParam dataSample[] = { -1,   -2,    1,    2,        0,
                            1.5, 2.5 , -1.5, -2.5, 1212.125};
  TypeParam labelsSample[] = {1,2};

  // end of manually updated part
  const size_t cols = this->cols_, rows = this->rows_;
  const size_t Dsize = sizeof(TypeParam), MB = 1000000;
  const float  chunk_size = 0.005;
  const size_t batch_size = MB / (cols * Dsize) * chunk_size;

  this->big_data_param_->set_chunk_size(chunk_size);

  BigDataLayer<TypeParam> layer(this->layer_param_);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  // at the begining .. allocate full `chunk_size` MB of space
  EXPECT_EQ( batch_size, this->top_data_->num() );
  EXPECT_EQ( this->top_label_->num(), this->top_data_->num() );
  EXPECT_EQ( 1, this->top_data_->channels() );
  EXPECT_EQ( 1, this->top_data_->height() );
  EXPECT_EQ( cols, this->top_data_->width() );
  EXPECT_EQ( 1, this->top_label_->channels() );
  EXPECT_EQ( 1, this->top_label_->height() );
  EXPECT_EQ( 1, this->top_label_->width() );

  // read once to test if the basic functionality works
  layer.Forward(this->bottom_vec_, this->top_vec_);
  for (int iter = 0; iter < (rows * 5); ++iter) {
    #ifdef DEBUG
    std::cout << "ids {r:"   << this->top_ids_->cpu_data()[iter]   << " ,e: " << iter % rows << "} "
              << "label {r:" << this->top_label_->cpu_data()[iter] << " ,e: " << labelsSample[iter % rows] << "} "
              << "data {r:"  << this->top_data_->cpu_data()[iter*cols] << ", "
                             << this->top_data_->cpu_data()[iter*cols+1] << ", "
                             << this->top_data_->cpu_data()[iter*cols+2] << ", "
                             << this->top_data_->cpu_data()[iter*cols+3] << ", "
                             << this->top_data_->cpu_data()[iter*cols+4] << "; "
              << "e: " << dataSample[(iter%rows)*cols] << ", "
                         << dataSample[(iter%rows)*cols+1] << ", "
                         << dataSample[(iter%rows)*cols+2] << ", "
                         << dataSample[(iter%rows)*cols+3] << ", "
                         << dataSample[(iter%rows)*cols+4] << "}"
              << std::endl;
    #endif
    // test iterations (iter) in order to see the cycling in reading a source file
    EXPECT_EQ(iter % rows, this->top_ids_->cpu_data()[iter]);
    EXPECT_EQ(labelsSample[iter % rows], this->top_label_->cpu_data()[iter]);
    for (int j = 0; j < cols; ++j) {
      EXPECT_EQ(dataSample[(iter%rows)*cols+j], this->top_data_->cpu_data()[(iter*cols)+j]);
    }
  }
  // read again to test cyclability of the source
  layer.Forward(this->bottom_vec_, this->top_vec_);
  // we don't know which ID will appear first - but it must match its data
  int id = this->top_ids_->cpu_data()[0];
  EXPECT_EQ(labelsSample[id], this->top_label_->cpu_data()[0]);
  for (int j = 0; j < cols; ++j) {
    EXPECT_EQ(dataSample[id*cols+j], this->top_data_->cpu_data()[j]);
  }
}


TYPED_TEST(BigDataLayerTest, TestSolelyIDs) {
  const TypeParam *ids;
  this->top_vec_.clear();
  this->top_vec_.push_back(this->top_data_);
  // this->top_vec_.push_back(this->top_label_); // leave it out on purpose
  this->top_vec_.push_back(this->top_ids_);

  BigDataLayer<TypeParam> layer(this->layer_param_);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  layer.Forward(this->bottom_vec_, this->top_vec_);
  ids = this->top_ids_->cpu_data();
  for(int i=0; i < std::min(this->top_ids_->count(0,1), 13); ++i) {
    // IDs are 1-based
    EXPECT_EQ(ids[i], 1 + (i % 2));
  }
}


TYPED_TEST(BigDataLayerTest, TestCache) {
  this->big_data_param_->set_cache(::caffe::BigDataParameter_CacheControl_ENABLED);

  BigDataLayer<TypeParam> layer(this->layer_param_);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  layer.Forward(this->bottom_vec_, this->top_vec_);
  std::ifstream stream;
  // at this point, a cache file should have been created
  stream.open(this->filename_ + string("bin"), std::ios::binary);
  EXPECT_EQ(true, stream.good());
  stream.close();

  stream.open(this->filename_ + string("bin.part"), std::ios::binary);
  EXPECT_EQ(false, stream.good());
  stream.close();

  remove((this->filename_ + string("bin")).c_str());
}



}  // namespace caffe
