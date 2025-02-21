@doc "
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"

module MNIST_loader

using PyCall

#### Libraries
# Standard library
@pyimport cPickle
@pyimport gzip

# Third-party libraries
#import numpy as np

function load_data()
    "Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    "
    f = gzip.open("../data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = cPickle.load(f)
    #f.close()
    return (training_data, validation_data, test_data)
end

columns(M) = [ slice(M,:,i) for i in 1:size(M, 2) ]
rows(M) = [ slice(M,i,:) for i in 1:size(M, 1) ]

function load_data_wrapper()
    "Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."
    tr_d, va_d, te_d = load_data()
    #training_inputs = [reshape(x, (784, 1)) for x in tr_d[1]]
    training_inputs = rows(tr_d[1])
    training_results = [vectorized_result(y) for y in tr_d[2]]
    training_data = collect(zip(training_inputs, training_results))
    #validation_inputs = [reshape(x, (784, 1)) for x in va_d[1]]
    validation_inputs = rows(va_d[1])
    validation_data = zip(validation_inputs, va_d[2])
    #test_inputs = [reshape(x, (784, 1)) for x in te_d[1]]
    test_inputs = rows(te_d[1])
    test_data = zip(test_inputs, te_d[2])
    return (training_data, validation_data, test_data)
end

function vectorized_result(j)
    "Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."
    e = zeros((10, 1))
    e[j+1] = 1.0
    return e
end

end
