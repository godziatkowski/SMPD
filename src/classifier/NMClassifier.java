package classifier;

import Jama.Matrix;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static probesUtilities.MatrixUtilities.getProbe;
import static probesUtilities.MatrixUtilities.pseudoinverseMoorePenrose;

public class NMClassifier implements IClassifier {

    private Map<String, double[][]> trainingSets;
    private Map<String, double[][]> testSets;
    private Set<String> keysInTrainingSets;
    private Set<String> keysInTestSets;

    private void getNamesOfTrainingAndTestSets() {
        keysInTestSets = testSets.keySet();
        keysInTrainingSets = trainingSets.keySet();
    }

    private void splitPassedSetsIntoTestAndTraining( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets ) {
        trainingSets = new HashMap<>();
        testSets = new HashMap<>();

        probesSplitedIntoTrainingAndTestSets.keySet().forEach( className -> {
            if ( className.contains( "_test" ) ) {
                testSets.put( className, probesSplitedIntoTrainingAndTestSets.get( className ) );
            } else {
                trainingSets.put( className, probesSplitedIntoTrainingAndTestSets.get( className ) );
            }
        } );
        getNamesOfTrainingAndTestSets();
    }

    private Map<String, double[]> calculateMeanValues( Map<String, double[][]> trainingSets ) {

        Map<String, double[]> meanAttributesPerClass = new HashMap<>();
        trainingSets.entrySet()
                .forEach( entry -> {
                    double[] means = new double[entry.getValue().length];
                    for ( int attributeIndex = 0; attributeIndex < entry.getValue().length; attributeIndex++ ) {
                        double mean = 0.0;
                        for ( int probeIndex = 0; probeIndex < entry.getValue()[attributeIndex].length; probeIndex++ ) {
                            mean += entry.getValue()[attributeIndex][probeIndex];
                        }
                        mean = mean / entry.getValue()[attributeIndex].length;
                        means[attributeIndex] = mean;
                    }
                    meanAttributesPerClass.put( entry.getKey(), means );
                } );

        return meanAttributesPerClass;
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

    private Map<String, double[][]> calculateCovarianceMatrix( Map<String, double[][]> probesWithSelectedAttributes, Map<String, double[]> meanValuesForEachClass ) {
        Map<String, double[][]> covarianceMatrixesForEachClass = new HashMap<>();

        probesWithSelectedAttributes.keySet().stream().forEach( className -> {
            double[][] probesInClass = probesWithSelectedAttributes.get( className );
            int countOfAttributes = probesInClass.length;
            int countOfProbes = probesInClass[0].length;
            double[][] covarianceMatrix = new double[countOfAttributes][countOfAttributes];
            double[] meanValuesForEachAttribute = meanValuesForEachClass.get( className );
            for ( int firstAttributeIterator = 0; firstAttributeIterator < countOfAttributes; firstAttributeIterator++ ) {
                for ( int secondAttributeIterator = 0; secondAttributeIterator < countOfAttributes; secondAttributeIterator++ ) {
                    double covarianceValue = 0.0;
                    for ( int probeIndex = 0; probeIndex < countOfProbes; probeIndex++ ) { //iteration over probes - we assume that every probe have all attributes
                        covarianceValue += ( probesInClass[firstAttributeIterator][probeIndex] - meanValuesForEachAttribute[firstAttributeIterator] ) * ( probesInClass[secondAttributeIterator][probeIndex] - meanValuesForEachAttribute[secondAttributeIterator] );
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
            }
            covarianceMatrixesForEachClass.put( className, matrix.getArray() );
        } );

        return covarianceMatrixesForEachClass;
    }

    @Override
    public double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes ) {
        Map<String, double[][]> probesSplitedIntoTrainingAndTestSetsWithExtreactedAttributes = extractSelectedAttributes( probesSplitedIntoTrainingAndTestSets, indexesOfBestAttributes.stream().collect( Collectors.toList() ) );
        splitPassedSetsIntoTestAndTraining( probesSplitedIntoTrainingAndTestSetsWithExtreactedAttributes );
        double percentageAlgorithmCorrectness = 0.0;
        int countOfFailedClassifications = 0;
        int countOfSuccessClassifications = 0;

        Map<String, double[]> meanAttributesPerClass = calculateMeanValues( trainingSets );
        Map<String, double[][]> covarianceMatrixes = calculateCovarianceMatrix( trainingSets, meanAttributesPerClass );

        for ( String keyInTestSet : keysInTestSets ) {
            double[][] probesInTestSet = testSets.get( keyInTestSet );
            for ( int probeIndexInTestSet = 0; probeIndexInTestSet < probesInTestSet[0].length; probeIndexInTestSet++ ) {
                Map<String, Double> distanceBetweenProbeAndCLassCentroid = new HashMap<>();
                double[] probe = getProbe( probesInTestSet, probeIndexInTestSet );
                Matrix probeAsMatrix = new Matrix( probe, 1 );
                keysInTrainingSets.stream().forEach( ( keyInTrainingSet ) -> {
                    Matrix covarianceMatrix = new Matrix( covarianceMatrixes.get( keyInTrainingSet ) );
                    Matrix meanMatrix = new Matrix( meanAttributesPerClass.get( keyInTrainingSet ), 1 );
                    Matrix probeMinusMean = probeAsMatrix.minus( meanMatrix );
                    double distanceToSet = probeMinusMean.times( covarianceMatrix ).times( probeMinusMean.transpose() ).get( 0, 0 );
                    distanceBetweenProbeAndCLassCentroid.put( keyInTrainingSet, distanceToSet );
                } );
                double smallestDistance = 1000000.0;
                String closestClass = "";
                Set<String> keySet = distanceBetweenProbeAndCLassCentroid.keySet();
                for ( String key : keySet ) {
                    if ( distanceBetweenProbeAndCLassCentroid.get( key ) < smallestDistance ) {
                        smallestDistance = distanceBetweenProbeAndCLassCentroid.get( key );
                        closestClass = key;
                    }
                }
                if ( keyInTestSet.contains( closestClass.replace( "_training", "" ).trim() ) ) {
                    countOfSuccessClassifications++;
                } else {
                    countOfFailedClassifications++;
                }
            }
        }

        percentageAlgorithmCorrectness = ( (double) countOfSuccessClassifications / (double) ( countOfFailedClassifications + countOfSuccessClassifications ) ) * 100;

        return percentageAlgorithmCorrectness;
    }

}
