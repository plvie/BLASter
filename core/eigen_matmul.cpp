#include <Eigen/Core>
#include <Eigen/Dense>

#include "types.hpp"

typedef Eigen::Matrix<ZZ, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXZZ;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynStride;

/**
 * Computes the matrix product between a and b, and stores the result a*b in c.
 * The dimensions of `a`, `b` and `c` should be n x m, m x k and n x k
 * respectively.
 *
 * Relevant reference:
 * https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
 */
void eigen_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k) {
	Eigen::Map<const MatrixXZZ> ma(a, n, m);
	Eigen::Map<const MatrixXZZ> mb(b, m, k);
	Eigen::Map<MatrixXZZ> mc(c, n, k);

	mc = ma * mb;
}

/**
 * Computes the matrix product between a and b, and stores the result a*b in a.
 * The dimensions of `a` and `b` should both be n x n.
 *
 * Relevant reference:
 * https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
 */
void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m) {
	Eigen::Map<MatrixXZZ> ma(a, n, m);
	Eigen::Map<const MatrixXZZ> mb(b, m, m);

	ma *= mb;
}


void eigen_right_matmul_strided(ZZ *a, const ZZ *b, int n, int m, int stride_a) {
	Eigen::Map<MatrixXZZ, Eigen::Unaligned, DynStride> ma(a, n, m, DynStride(stride_a, 1));
	Eigen::Map<const MatrixXZZ> mb(b, m, m);

	ma *= mb;
}


void eigen_init(int num_cores) {
	// See https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
	Eigen::initParallel();
	Eigen::setNbThreads(num_cores);
}
