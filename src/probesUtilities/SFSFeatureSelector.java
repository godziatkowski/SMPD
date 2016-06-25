package probesUtilities;

import java.util.ArrayList;
import java.util.List;

public class SFSFeatureSelector {

    public List<Integer> selectBestFeatureIndexesUsingSFS( int numberOfFeaturesToSelect, double[][] probes, int firstBestFeatureIndex ) {
        List<Integer> bestFeatureIndexes = new ArrayList<>( numberOfFeaturesToSelect );

//        double tmp;
//        
//        bestFeatureIndexes.add( firstBestFeatureIndex );
//
//        for ( int i = 1; i < numberOfFeaturesToSelect; i++ ) {
//            double fisherDiscriminant = Double.MIN_VALUE;
//            bestFeatureIndexes.add( -1 );
//
//            for ( int j = 0; j < probes.length; j++ ) {
//                if ( bestFeatureIndexes.contains( j ) ) {
//                    continue;
//                }
//
//                int[] featureIndexes = new int[i + 1];
//                for ( int k = 0; k < i; k++ ) {
//                    featureIndexes[k] = bestFeatureIndexes.get( k );
//                }
//                featureIndexes[i] = j;
//
//                tmp = computeFisherLD( probes, featureIndexes );
//                if ( tmp > fisherDiscriminant ) {
//                    fisherDiscriminant = tmp;
//                    bestFeatureIndexes.set( i, j );
//                }
//            }
//        }        

        return bestFeatureIndexes;
    }

    

}
