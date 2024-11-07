#include <Eigen/Core>
#include <Eigen/Dense>

#include "types.hpp"


template<typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
typedef Eigen::Stride<Eigen::Dynamic, 1> Stride;


void eigen_init(int num_cores) {
	// See https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
	Eigen::initParallel();
	Eigen::setNbThreads(num_cores);
}


// See: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `c`.
 * Dimensions of `a`, `b` and `c` are assumed to be `n x m`, `m x k` and `n x k` respectively.
 */
template<typename T>
void eigen_matmul(const T *a, const T *b, T *c, int n, int m, int k) {
	Eigen::Map<const Matrix<T>> ma(a, n, m), mb(b, m, k);
	Eigen::Map<Matrix<T>> mc(c, n, k);

	mc = ma * mb;
}

template<typename T>
void eigen_matmul(const T *a, const T *b, T *c, int n, int m, int k, int stride_a) {
	Eigen::Map<const Matrix<T>, Eigen::Unaligned, Stride> ma(a, n, m, Stride(stride_a, 1));
	Eigen::Map<const Matrix<T>> mb(b, m, k);
	Eigen::Map<Matrix<T>> mc(c, n, k);

	mc = ma * mb;
}

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `a`.
 * Dimensions of `a` and `b` are assumed to be `n x m` and `m x m` respectively.
 */
template<typename T>
void eigen_right_matmul(T *a, const T *b, int n, int m) {
	Eigen::Map<Matrix<T>> ma(a, n, m);
	Eigen::Map<const Matrix<T>> mb(b, m, m);

	ma *= mb;
}

template<typename T>
void eigen_right_matmul(T *a, const T *b, int n, int m, int stride_a) {
	Eigen::Map<Matrix<T>, Eigen::Unaligned, Stride> ma(a, n, m, Stride(stride_a, 1));
	Eigen::Map<const Matrix<T>> mb(b, m, m);

	ma *= mb;
}
