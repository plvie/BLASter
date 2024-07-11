#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXZZ;

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
void eigen_right_matmul(long long *a, const long long *b, int n) {
	Eigen::Map<MatrixXZZ> ma(a, n, n);
	Eigen::Map<const MatrixXZZ> mb(b, n, n);

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
