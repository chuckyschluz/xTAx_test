#include "Eigen/Eigen"
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;
using Scalar_t = double;
using Int_t = int;
using Sparse_t = SparseMatrix<Scalar_t, ColMajor, Int_t>;
using VectorX_t = VectorX<Scalar_t>;

Scalar_t xTAx(const Sparse_t& A, const VectorX_t& x)
{
	Scalar_t result = 0;
	int k = 0;
	for (int j = 0; j < A.outerSize(); ++j)
		while (k < A.outerIndexPtr()[j + 1])
		{
			int i = A.innerIndexPtr()[k];
			Scalar_t value = x.coeff(i) * A.valuePtr()[k] * x.coeff(j);
			result += (i == j) ? (value) : (2 * value);
			++k;
		}
	return result;
}

Scalar_t xTAx_2(const Sparse_t& A, const VectorX_t& x)
{
	Scalar_t result = 0;
	int k = 0;

	for (int j = 0; j < A.outerSize(); ++j)
	{
		bool non_empty_column = k < A.outerIndexPtr()[j + 1];
		if (non_empty_column)
		{
			result += x.coeff(j) * A.valuePtr()[k] * x.coeff(j);//diag
			for (++k; k < A.outerIndexPtr()[j + 1]; ++k)
			{
				int i = A.innerIndexPtr()[k];
				result += 2 * x.coeff(i) * A.valuePtr()[k] * x.coeff(j);//non-diag
			}
		}
	}
	return result;
}

int main()
{
	Sparse_t Z;
	Eigen::loadMarket(Z, "Z_0.mtx");
	VectorX_t x = VectorX_t::Random(Z.rows());
	Scalar_t result = 0;

	auto start = std::chrono::high_resolution_clock::now();

	for (int iter = 0; iter < 2000000000; iter++)
		result += xTAx_2(Z, x);
		
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "One step duration: " << duration.count() << "\n";
	std::cout << result << "\n";
}