
package probesUtilities;

import Jama.Matrix;


public class EuklideanDistance {
    public static double calculateEuklideanDistanceBetweenMatrixes( Matrix firstMatrix, Matrix secondMatrix ) {
        Matrix matrixAfterSubtraction = firstMatrix.minus( secondMatrix );
        return Math.sqrt( matrixAfterSubtraction.times( matrixAfterSubtraction.transpose() ).get( 0, 0 ) );
    }

}
