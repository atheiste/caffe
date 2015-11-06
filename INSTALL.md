# Installation

See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.

Check the users group in case you need help:
https://groups.google.com/forum/#!forum/caffe-users


# Testing
To run all tests type

  $ make runtest

It is possible to separate phases of compilation and running. To compile the test use

  $ make test

If you plan to debug then use

  $ make all -e DEBUG=1

in order to have debug symbols in the binaries.

Tests are represented by runners, which are Google test binaries. Google test framework
which provides few useful options. The most used is to filter tests by name. For
example we want to run all tests with CPU in the name

  $ test/test.testbin --gtest_filter='*CPU*'

or to run one specific test, you can run it directly from the test folder

  $ build/test/test_<your-test>.testbin

Please note that files, used in tests, are available on relative path to your
CWD in the moment of running tests.
