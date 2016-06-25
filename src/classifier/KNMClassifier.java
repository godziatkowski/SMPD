package classifier;

import Jama.Matrix;
import exceptions.CannotInverseMatrixException;
import java.util.*;
import java.util.stream.Collectors;
import probesUtilities.ClusterFinder;

import static probesUtilities.MatrixUtilities.getProbe;
import static probesUtilities.MatrixUtilities.pseudoinverseMoorePenrose;

public class KNMClassifier implements IClassifier {

    private final ClusterFinder clusterFinder;

    private Set<String> testSetsKeys;
    private Set<String> trainingSetsKeys;

    public KNMClassifier() {
        clusterFinder = new ClusterFinder();
    }

    private Map<String, double[][]> extractSelectedAttributes( Map<String, double[][]> probesWithAllAttributes, List<Integer> indexesOfBestAttributes ) {
        Map<String, double[][]> probesWithExtractedAttributes = new HashMap<>();
        probesWithAllAttributes.keySet().stream().forEach( ( className ) -> {
            double[][] probesInSet = probesWithAllAttributes.get( className );
            double[][] probesInSetWithExtractedAttributes = new double[indexesOfBestAttributes.size()][probesInSet[0].length];
            int index = 0;
            for ( Integer indexOfBestAttribute : indexesOfBestAttributes ) {
                probesInSetWithExtractedAttributes[index] = probesInSet[indexOfBestAttribute];
                index++;
            }
            probesWithExtractedAttributes.put( className, probesInSetWithExtractedAttributes );
        } );

        return probesWithExtractedAttributes;
    }

    private double[][] calculateCovarianceMatrix( double[][] probes, double[] clusterCentroid ) throws CannotInverseMatrixException {
        int countOfAttributes = clusterCentroid.length;
        int countOfProbes = probes[0].length;
        double[][] covarianceMatrix = new double[countOfAttributes][countOfAttributes];
        double[] meanValuesForEachAttribute = clusterCentroid;
        for ( int firstAttributeIterator = 0; firstAttributeIterator < countOfAttributes; firstAttributeIterator++ ) {
            for ( int secondAttributeIterator = 0; secondAttributeIterator < countOfAttributes; secondAttributeIterator++ ) {
                double covarianceValue = 0.0;
                for ( int probeIndex = 0; probeIndex < countOfProbes; probeIndex++ ) { //iteration over probes - we assume that every probe have all attributes
                    covarianceValue += ( probes[firstAttributeIterator][probeIndex] - clusterCentroid[firstAttributeIterator] ) * ( probes[secondAttributeIterator][probeIndex] - meanValuesForEachAttribute[secondAttributeIterator] );
                }
                covarianceValue = covarianceValue / ( countOfProbes - 1 );
                covarianceMatrix[firstAttributeIterator][secondAttributeIterator] = covarianceValue;
            }
        }
        Matrix matrix = new Matrix( covarianceMatrix );
        try {
            matrix = matrix.inverse();
        } catch ( Exception e ) {
            matrix = pseudoinverseMoorePenrose( matrix );
            if ( matrix == null ) {
                throw new CannotInverseMatrixException();
            }
        }
        return matrix.getArray();
    }

    @Override
    public double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> bestAttributes ) {

        int countOfFailedClassifications = 0;
        int countOfSuccessClassifications = 0;
        probesSplitedIntoTrainingAndTestSets = extractSelectedAttributes( probesSplitedIntoTrainingAndTestSets, bestAttributes.stream().collect( Collectors.toList() ) );

        trainingSetsKeys = probesSplitedIntoTrainingAndTestSets.keySet()
                .stream()
                .filter( key -> key.contains( "_training" ) )
                .collect( Collectors.toSet() );
        testSetsKeys = probesSplitedIntoTrainingAndTestSets.keySet()
                .stream()
                .filter( key -> key.contains( "_test" ) )
                .collect( Collectors.toSet() );
        Map<String, Map<Matrix, Matrix>> modsWithAssignedProbesForEachClass = new HashMap<>();
        for ( String trainingSetsKey : trainingSetsKeys ) {
            double[][] probes = probesSplitedIntoTrainingAndTestSets.get( trainingSetsKey );
            modsWithAssignedProbesForEachClass.put( trainingSetsKey, clusterFinder.assignProbesToClusters( new Matrix( probes ) ) );

        }

        for ( String testSetKey : testSetsKeys ) {
            double[][] probesInTestSet = probesSplitedIntoTrainingAndTestSets.get( testSetKey );
            for ( int probeIndex = 0; probeIndex < probesInTestSet[0].length; probeIndex++ ) {
                Map<String, Double> distancesBetweenProbeAndClass = new HashMap<>();
                Matrix probeAsMatrix = new Matrix( getProbe( probesInTestSet, probeIndex ), 1 );
                for ( String trainingSetName : trainingSetsKeys ) {
                    while ( true ) {
                        try {
                            double distanceToNearestClassCentroid = Double.MAX_VALUE;
                            for ( Map.Entry<Matrix, Matrix> entrySet : modsWithAssignedProbesForEachClass.get( trainingSetName ).entrySet() ) {
                                Matrix covarianceMatrix = new Matrix( calculateCovarianceMatrix( entrySet.getValue().getArray(), entrySet.getKey().getRowPackedCopy() ) );
                                if ( !( covarianceMatrix.det() == 0 ) ) {
                                    Matrix probeMinusMean = probeAsMatrix.minus( entrySet.getKey() );

                                    double distanceToSet = probeMinusMean.times( covarianceMatrix ).times( probeMinusMean.transpose() ).get( 0, 0 );
                                    if ( distanceToSet < distanceToNearestClassCentroid ) {
                                        distanceToNearestClassCentroid = distanceToSet;
                                    }
                                }
                            }
                            distancesBetweenProbeAndClass.put( trainingSetName, distanceToNearestClassCentroid );
                            break;
                        } catch ( CannotInverseMatrixException e ) {
                        }
                    }
                }

                double smallestDistance = Double.MAX_VALUE;
                String closestClass = "";
                for ( String trainingSetKey : trainingSetsKeys ) {
                    if ( distancesBetweenProbeAndClass.get( trainingSetKey ) < smallestDistance ) {
                        smallestDistance = distancesBetweenProbeAndClass.get( trainingSetKey );
                        closestClass = trainingSetKey;
                    }
                }
                if ( testSetKey.contains( closestClass.replace( "_training", "" ).trim() ) ) {
                    countOfSuccessClassifications++;
                } else {
                    countOfFailedClassifications++;
                }
            }
        }

        double percentageAlgorithmCorrectness = ( (double) countOfSuccessClassifications / (double) ( countOfFailedClassifications + countOfSuccessClassifications ) ) * 100;
        return percentageAlgorithmCorrectness;
    }

    private void printMatrix( Matrix matrix ) {
        double[][] matrixAsArray = matrix.getArray();
        for ( double[] row : matrixAsArray ) {
            printArrayOneDim( row );
        }
    }

    private void printArrayOneDim( double[] array ) {
        String row = "";
        for ( double element : array ) {
            row += element + " ";
        }
        System.out.println( row );
    }

}
