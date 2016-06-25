package probesUtilities;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class MatrixUtilities {

    private static final double MACHEPS = 2E-16;

    public static double[] getProbe( double[][] matrix, int probeIndex ) {
        double[] probe = new double[matrix.length];
        for ( int attributeIndex = 0; attributeIndex < matrix.length; attributeIndex++ ) {
            probe[attributeIndex] = matrix[attributeIndex][probeIndex];
        }
        return probe;
    }

    /**
     * Computes the Moore�Penrose pseudoinverse using the SVD method. copied
     * from
     * http://the-lost-beauty.blogspot.com/2009/04/moore-penrose-pseudoinverse-in-jama.html
     */
    public static Matrix pseudoinverseMoorePenrose( Matrix x ) {
        int rows = x.getRowDimension();
        int cols = x.getColumnDimension();
        if ( rows < cols ) {
            Matrix result = pseudoinverseMoorePenrose( x.transpose() );
            if ( result != null ) {
                result = result.transpose();
            }
            return result;
        }
        SingularValueDecomposition svdX = new SingularValueDecomposition( x );
        if ( svdX.rank() < 1 ) {
            return null;
        }
        double[] singularValues = svdX.getSingularValues();
        double tol = Math.max( rows, cols ) * singularValues[0] * MACHEPS;
        double[] singularValueReciprocals = new double[singularValues.length];
        for ( int i = 0; i < singularValues.length; i++ ) {
            if ( Math.abs( singularValues[i] ) >= tol ) {
                singularValueReciprocals[i] = 1.0 / singularValues[i];
            }
        }
        double[][] u = svdX.getU().getArray();
        double[][] v = svdX.getV().getArray();
        int min = Math.min( cols, u[0].length );
        double[][] inverse = new double[cols][rows];
        for ( int i = 0; i < cols; i++ ) {
            for ( int j = 0; j < u.length; j++ ) {
                for ( int k = 0; k < min; k++ ) {
                    inverse[i][j] += v[i][k] * singularValueReciprocals[k]
                                     * u[j][k];
                }
            }
        }
        return new Matrix( inverse );
    }

}
