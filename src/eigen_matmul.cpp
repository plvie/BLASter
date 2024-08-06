#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXZZ;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynStride;

/**
 * Computes the matrix product between a and b, and stores the result a*b in c.
 * The dimensions of `a`, `b` and `c` should be n x m, m x k and n x k
 * respectively.
 *
 * Relevant reference:
 * https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
 */
void eigen_matmul(const long long *a, const long long *b, long long *c, int n, int m, int k) {
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
void eigen_right_matmul(long long *a, const long long *b, int n, int m) {
	Eigen::Map<MatrixXZZ> ma(a, n, m);
	Eigen::Map<const MatrixXZZ> mb(b, m, m);

	ma *= mb;
}


void eigen_right_matmul_strided(long long *a, const long long *b, int n, int m, int stride_a, int stride_b) {
	Eigen::Map<MatrixXZZ, Eigen::Unaligned, DynStride> ma(a, n, m, DynStride(stride_a, 1));
	Eigen::Map<const MatrixXZZ, Eigen::Unaligned, DynStride> mb(b, m, m, DynStride(stride_b, 1));

	ma *= mb;
}


void eigen_init(int num_cores) {
	Eigen::initParallel();

	if (num_cores > 0) {
		Eigen::setNbThreads(num_cores);
	}

	// To check number of threads, use:
	// printf("%d ", (int) Eigen::nbThreads());
}
